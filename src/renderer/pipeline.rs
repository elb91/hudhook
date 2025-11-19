use std::collections::HashMap;
use std::mem;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use imgui::Context;
use once_cell::sync::{Lazy, OnceCell};
use parking_lot::Mutex;
use tracing::error;
use windows::core::{Error, Result, HRESULT};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    CallWindowProcW, DefWindowProcW, SetWindowLongPtrW, GWLP_WNDPROC,
    WM_CHAR, WM_KEYDOWN, WM_KEYUP, WM_LBUTTONDOWN, WM_LBUTTONUP,
    WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEMOVE, WM_MOUSEWHEEL,
    WM_RBUTTONDOWN, WM_RBUTTONUP,
};

use crate::renderer::input::{imgui_wnd_proc_impl, WndProcType};
use crate::renderer::RenderEngine;
use crate::{util, ImguiRenderLoop, MessageFilter};

type RenderLoop = Box<dyn ImguiRenderLoop + Send + Sync>;

static PIPELINE_STATES: Lazy<Mutex<HashMap<isize, Arc<PipelineSharedState>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug)]
pub(crate) struct PipelineMessage(
    pub(crate) HWND,
    pub(crate) u32,
    pub(crate) WPARAM,
    pub(crate) LPARAM,
);

pub(crate) struct PipelineSharedState {
    pub(crate) message_filter: AtomicU32,
    pub(crate) wnd_proc: WndProcType,
    pub(crate) tx: Sender<PipelineMessage>,
}

pub(crate) struct Pipeline<T: RenderEngine> {
    hwnd: HWND,
    ctx: Context,
    engine: T,
    render_loop: RenderLoop,
    rx: Receiver<PipelineMessage>,
    shared_state: Arc<PipelineSharedState>,
    queue_buffer: OnceCell<Vec<PipelineMessage>>,
    start_of_first_frame: OnceCell<Instant>,
}

impl<T: RenderEngine> Pipeline<T> {
    pub(crate) fn new(
        hwnd: HWND,
        mut ctx: Context,
        mut engine: T,
        mut render_loop: RenderLoop,
    ) -> std::result::Result<Self, (Error, RenderLoop)> {
        let (width, height) = util::win_size(hwnd);

        ctx.io_mut().display_size = [width as f32, height as f32];

        render_loop.initialize(&mut ctx, &mut engine);

        if let Err(e) = engine.setup_fonts(&mut ctx) {
            return Err((e, render_loop));
        }

        let wnd_proc = unsafe {
            #[cfg(target_arch = "x86")]
            type SwlpRet = i32;
            #[cfg(target_arch = "x86_64")]
            type SwlpRet = isize;

            mem::transmute::<SwlpRet, WndProcType>(SetWindowLongPtrW(
                hwnd,
                GWLP_WNDPROC,
                pipeline_wnd_proc as usize as _,
            ))
        };

        let (tx, rx) = mpsc::channel();
        let shared_state = Arc::new(PipelineSharedState {
            // Start with blocking all input by default
            // This prevents race condition where messages arrive before first frame
            message_filter: AtomicU32::new(MessageFilter::all().bits()),
            wnd_proc,
            tx,
        });

        PIPELINE_STATES.lock().insert(hwnd.0, Arc::clone(&shared_state));

        let queue_buffer = OnceCell::from(Vec::new());

        Ok(Self {
            hwnd,
            ctx,
            engine,
            render_loop,
            rx,
            shared_state: Arc::clone(&shared_state),
            queue_buffer,
            start_of_first_frame: OnceCell::new(),
        })
    }

    pub(crate) fn prepare_render(&mut self) -> Result<()> {
        // Process messages BEFORE updating filter
        // This ensures we process messages from previous frame before making new decisions
        let mut queue_buffer = self.queue_buffer.take().unwrap();
        queue_buffer.clear();
        queue_buffer.extend(self.rx.try_iter());
        queue_buffer.drain(..).for_each(|PipelineMessage(hwnd, umsg, wparam, lparam)| {
            imgui_wnd_proc_impl(hwnd, umsg, wparam, lparam, self);
        });
        self.queue_buffer.set(queue_buffer).expect("OnceCell should be empty");

        // Now update message filter for THIS frame
        let message_filter = self.render_loop.message_filter(self.ctx.io());
        self.shared_state.message_filter.store(message_filter.bits(), Ordering::SeqCst);

        let io = self.ctx.io_mut();
        io.nav_active = true;
        io.nav_visible = true;

        self.render_loop.before_render(&mut self.ctx, &mut self.engine);

        Ok(())
    }

    pub(crate) fn render(&mut self, render_target: T::RenderTarget) -> Result<()> {
        let delta_time = Instant::now()
            .checked_duration_since(*self.start_of_first_frame.get_or_init(Instant::now))
            .unwrap_or(Duration::ZERO)
            .checked_sub(Duration::from_secs_f64(self.ctx.time()))
            .unwrap_or(Duration::ZERO);

        self.ctx.io_mut().update_delta_time(delta_time);

        let [w, h] = self.ctx.io().display_size;
        let [fsw, fsh] = self.ctx.io().display_framebuffer_scale;

        if (w * fsw) <= 0.0 || (h * fsh) <= 0.0 {
            error!("Insufficient display size: {w}x{h}");
            return Err(Error::from_hresult(HRESULT(-1)));
        }

        let ui = self.ctx.frame();
        self.render_loop.render(ui);
        let draw_data = self.ctx.render();

        self.engine.render(draw_data, render_target)?;

        Ok(())
    }

    pub(crate) fn context(&mut self) -> &mut Context {
        &mut self.ctx
    }

    pub(crate) fn render_loop(&mut self) -> &mut RenderLoop {
        &mut self.render_loop
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.ctx.io_mut().display_size = [width as f32, height as f32];
    }

    pub(crate) fn cleanup(&mut self) {
        unsafe {
            SetWindowLongPtrW(self.hwnd, GWLP_WNDPROC, self.shared_state.wnd_proc as usize as _)
        };
    }

    pub(crate) fn take(mut self) -> RenderLoop {
        self.cleanup();
        self.render_loop
    }
}

// Conservative blocking approach
unsafe extern "system" fn pipeline_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let shared_state = {
        let Some(shared_state_guard) = PIPELINE_STATES.try_lock() else {
            error!("Could not lock shared state in window procedure");
            // Block input when we can't access state
            return match msg {
                WM_LBUTTONDOWN | WM_LBUTTONUP | WM_RBUTTONDOWN | WM_RBUTTONUP |
                WM_MBUTTONDOWN | WM_MBUTTONUP | WM_MOUSEWHEEL | WM_MOUSEMOVE |
                WM_KEYDOWN | WM_KEYUP | WM_CHAR => LRESULT(1),
                _ => DefWindowProcW(hwnd, msg, wparam, lparam),
            };
        };

        let Some(shared_state) = shared_state_guard.get(&hwnd.0) else {
            error!("Could not get shared state for handle {hwnd:?}");
            return DefWindowProcW(hwnd, msg, wparam, lparam);
        };

        Arc::clone(shared_state)
    };

    // Always queue the message for processing
    if let Err(e) = shared_state.tx.send(PipelineMessage(hwnd, msg, wparam, lparam)) {
        error!("Could not send window message through pipeline: {e:?}");
    }

    // Get current message filter
    let message_filter =
        MessageFilter::from_bits_retain(shared_state.message_filter.load(Ordering::SeqCst));

    // The filter is from the PREVIOUS frame, but that's OK
    // It's conservative - if UI wanted input last frame, block this frame too
    // Better to be slightly late blocking than to leak clicks to game
    if message_filter.is_blocking(msg) {
        LRESULT(1)
    } else {
        CallWindowProcW(Some(shared_state.wnd_proc), hwnd, msg, wparam, lparam)
    }
}
