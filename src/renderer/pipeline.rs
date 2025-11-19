use std::collections::HashMap;
use std::mem;
use std::sync::Arc;
use std::time::{Duration, Instant};

use imgui::Context;
use once_cell::sync::{Lazy, OnceCell};
use parking_lot::Mutex;
use tracing::{error, debug};
use windows::core::{Error, Result, HRESULT};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    CallWindowProcW, DefWindowProcW, SetWindowLongPtrW, GWLP_WNDPROC,
    WM_CHAR, WM_KEYDOWN, WM_KEYUP, WM_LBUTTONDOWN, WM_LBUTTONUP,
    WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEMOVE, WM_MOUSEWHEEL,
    WM_RBUTTONDOWN, WM_RBUTTONUP,
};

use crate::renderer::input::WndProcType;
use crate::renderer::RenderEngine;
use crate::{util, ImguiRenderLoop, MessageFilter};

type RenderLoop = Box<dyn ImguiRenderLoop + Send + Sync>;

// Shared state accessible from window procedure
static PIPELINE_STATES: Lazy<Mutex<HashMap<isize, Arc<PipelineSharedState>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// Store Context pointers (unsafe but controlled)
struct ContextPtr(*mut Context);
unsafe impl Send for ContextPtr {}
unsafe impl Sync for ContextPtr {}

static PIPELINE_CONTEXTS: Lazy<Mutex<HashMap<isize, ContextPtr>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub(crate) struct PipelineSharedState {
    pub(crate) wnd_proc: WndProcType,
}

pub(crate) struct Pipeline<T: RenderEngine> {
    hwnd: HWND,
    ctx: Context,
    engine: T,
    render_loop: RenderLoop,
    shared_state: Arc<PipelineSharedState>,
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

        let shared_state = Arc::new(PipelineSharedState { wnd_proc });

        PIPELINE_STATES.lock().insert(hwnd.0, Arc::clone(&shared_state));

        Ok(Self {
            hwnd,
            ctx,
            engine,
            render_loop,
            shared_state: Arc::clone(&shared_state),
            start_of_first_frame: OnceCell::new(),
        })
    }

    pub(crate) fn prepare_render(&mut self) -> Result<()> {
        // Store pointer to context for WndProc access
        let ctx_ptr = &mut self.ctx as *mut Context;
        PIPELINE_CONTEXTS.lock().insert(self.hwnd.0, ContextPtr(ctx_ptr));

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
        
        // Remove context pointer after render
        PIPELINE_CONTEXTS.lock().remove(&self.hwnd.0);

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
        PIPELINE_STATES.lock().remove(&self.hwnd.0);
        PIPELINE_CONTEXTS.lock().remove(&self.hwnd.0);
        
        unsafe {
            SetWindowLongPtrW(self.hwnd, GWLP_WNDPROC, self.shared_state.wnd_proc as usize as _)
        };
    }

    pub(crate) fn take(mut self) -> RenderLoop {
        self.cleanup();
        self.render_loop
    }
}

// Process messages synchronously
unsafe extern "system" fn pipeline_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // Get shared state for original wnd proc
    let shared_state = {
        let Some(shared_state_guard) = PIPELINE_STATES.try_lock() else {
            return DefWindowProcW(hwnd, msg, wparam, lparam);
        };

        let Some(shared_state) = shared_state_guard.get(&hwnd.0) else {
            return DefWindowProcW(hwnd, msg, wparam, lparam);
        };

        Arc::clone(shared_state)
    };

    // Try to get context pointer
    let ctx_ptr = PIPELINE_CONTEXTS.lock().get(&hwnd.0).map(|p| p.0);

    // If we have context, process message synchronously
    if let Some(ctx_ptr) = ctx_ptr {
        let ctx = &mut *ctx_ptr;
        
        // Update ImGui with this message
        process_imgui_message(ctx, msg, wparam, lparam);
        
        // Make decision with CURRENT state
        let io = ctx.io();
        let should_block = match msg {
            WM_LBUTTONDOWN | WM_LBUTTONUP | WM_RBUTTONDOWN | WM_RBUTTONUP |
            WM_MBUTTONDOWN | WM_MBUTTONUP | WM_MOUSEWHEEL | WM_MOUSEMOVE => {
                io.want_capture_mouse
            }
            WM_KEYDOWN | WM_KEYUP | WM_CHAR => {
                io.want_capture_keyboard
            }
            _ => false,
        };
        
        if should_block {
            return LRESULT(1);
        }
    } else {
        // Conservative: block input when context not available
        match msg {
            WM_LBUTTONDOWN | WM_LBUTTONUP | WM_RBUTTONDOWN | WM_RBUTTONUP |
            WM_MBUTTONDOWN | WM_MBUTTONUP | WM_MOUSEWHEEL | WM_MOUSEMOVE |
            WM_KEYDOWN | WM_KEYUP | WM_CHAR => {
                return LRESULT(1);
            }
            _ => {}
        }
    }

    // Forward to original window procedure
    CallWindowProcW(Some(shared_state.wnd_proc), hwnd, msg, wparam, lparam)
}

// Process Windows message and update ImGui
unsafe fn process_imgui_message(
    ctx: &mut Context,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) {
    let io = ctx.io_mut();
    
    match msg {
        WM_LBUTTONDOWN => {
            io.add_mouse_button_event(imgui::MouseButton::Left, true);
        }
        WM_LBUTTONUP => {
            io.add_mouse_button_event(imgui::MouseButton::Left, false);
        }
        WM_RBUTTONDOWN => {
            io.add_mouse_button_event(imgui::MouseButton::Right, true);
        }
        WM_RBUTTONUP => {
            io.add_mouse_button_event(imgui::MouseButton::Right, false);
        }
        WM_MBUTTONDOWN => {
            io.add_mouse_button_event(imgui::MouseButton::Middle, true);
        }
        WM_MBUTTONUP => {
            io.add_mouse_button_event(imgui::MouseButton::Middle, false);
        }
        WM_MOUSEWHEEL => {
            let wheel_delta = ((wparam.0 >> 16) & 0xffff) as i16 as f32 / 120.0;
            io.add_mouse_wheel_event([0.0, wheel_delta]);
        }
        WM_MOUSEMOVE => {
            let x = (lparam.0 & 0xffff) as i16 as f32;
            let y = ((lparam.0 >> 16) & 0xffff) as i16 as f32;
            io.add_mouse_pos_event([x, y]);
        }
        WM_KEYDOWN | WM_KEYUP => {
            let vk = wparam.0 as i32;
            let is_down = msg == WM_KEYDOWN;
            if vk < 256 {
                if let Some(key) = vk_to_imgui_key(vk) {
                    io.add_key_event(key, is_down);
                }
            }
        }
        WM_CHAR => {
            if wparam.0 > 0 && wparam.0 < 0x10000 {
                if let Some(c) = char::from_u32(wparam.0 as u32) {
                    io.add_input_character(c);
                }
            }
        }
        _ => {}
    }
}

// Convert VK codes to ImGui keys
fn vk_to_imgui_key(vk: i32) -> Option<imgui::Key> {
    use imgui::Key;
    Some(match vk {
        0x09 => Key::Tab,
        0x25 => Key::LeftArrow,
        0x26 => Key::UpArrow,
        0x27 => Key::RightArrow,
        0x28 => Key::DownArrow,
        0x21 => Key::PageUp,
        0x22 => Key::PageDown,
        0x24 => Key::Home,
        0x23 => Key::End,
        0x2D => Key::Insert,
        0x2E => Key::Delete,
        0x08 => Key::Backspace,
        0x20 => Key::Space,
        0x0D => Key::Enter,
        0x1B => Key::Escape,
        0x11 | 0xA2 => Key::LeftCtrl,
        0xA3 => Key::RightCtrl,
        0x10 | 0xA0 => Key::LeftShift,
        0xA1 => Key::RightShift,
        0x12 | 0xA4 => Key::LeftAlt,
        0xA5 => Key::RightAlt,
        0x5B => Key::LeftSuper,
        0x5C => Key::RightSuper,
        0x30 => Key::Alpha0,
        0x31 => Key::Alpha1,
        0x32 => Key::Alpha2,
        0x33 => Key::Alpha3,
        0x34 => Key::Alpha4,
        0x35 => Key::Alpha5,
        0x36 => Key::Alpha6,
        0x37 => Key::Alpha7,
        0x38 => Key::Alpha8,
        0x39 => Key::Alpha9,
        0x41 => Key::A,
        0x42 => Key::B,
        0x43 => Key::C,
        0x44 => Key::D,
        0x45 => Key::E,
        0x46 => Key::F,
        0x47 => Key::G,
        0x48 => Key::H,
        0x49 => Key::I,
        0x4A => Key::J,
        0x4B => Key::K,
        0x4C => Key::L,
        0x4D => Key::M,
        0x4E => Key::N,
        0x4F => Key::O,
        0x50 => Key::P,
        0x51 => Key::Q,
        0x52 => Key::R,
        0x53 => Key::S,
        0x54 => Key::T,
        0x55 => Key::U,
        0x56 => Key::V,
        0x57 => Key::W,
        0x58 => Key::X,
        0x59 => Key::Y,
        0x5A => Key::Z,
        0x70 => Key::F1,
        0x71 => Key::F2,
        0x72 => Key::F3,
        0x73 => Key::F4,
        0x74 => Key::F5,
        0x75 => Key::F6,
        0x76 => Key::F7,
        0x77 => Key::F8,
        0x78 => Key::F9,
        0x79 => Key::F10,
        0x7A => Key::F11,
        0x7B => Key::F12,
        _ => return None,
    })
}
