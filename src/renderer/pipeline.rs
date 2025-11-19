use std::collections::HashMap;
use std::mem;
use std::sync::atomic::{AtomicU32, Ordering};
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

use crate::renderer::input::{imgui_wnd_proc_impl, WndProcType};
use crate::renderer::RenderEngine;
use crate::{util, ImguiRenderLoop, MessageFilter};

type RenderLoop = Box<dyn ImguiRenderLoop + Send + Sync>;

// Store only the window procedure and original wnd proc
static PIPELINE_STATES: Lazy<Mutex<HashMap<isize, Arc<PipelineSharedState>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// Separate storage for mutable pipeline access
static PIPELINES: Lazy<Mutex<HashMap<isize, *mut PipelineData>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub(crate) struct PipelineSharedState {
    pub(crate) wnd_proc: WndProcType,
}

// Internal mutable data stored separately
struct PipelineData {
    ctx: Context,
    message_filter: MessageFilter,
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
        // Store pointer to our context for WndProc to access
        let data = Box::new(PipelineData {
            ctx: unsafe { std::ptr::read(&self.ctx) }, // Unsafe: temporarily move out
            message_filter: MessageFilter::empty(),
        });
        
        let data_ptr = Box::into_raw(data);
        PIPELINES.lock().insert(self.hwnd.0, data_ptr);

        // Get message filter from render loop
        let message_filter = self.render_loop.message_filter(self.ctx.io());
        
        // Update the stored message filter
        unsafe {
            (*data_ptr).message_filter = message_filter;
        }

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
        
        // Clean up the pipeline data pointer after render
        if let Some(data_ptr) = PIPELINES.lock().remove(&self.hwnd.0) {
            unsafe {
                // Move context back
                std::ptr::write(&mut self.ctx, (*data_ptr).ctx);
                // Free the box
                let _ = Box::from_raw(data_ptr);
            }
        }

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
        // Remove from both maps
        PIPELINE_STATES.lock().remove(&self.hwnd.0);
        if let Some(data_ptr) = PIPELINES.lock().remove(&self.hwnd.0) {
            unsafe {
                let _ = Box::from_raw(data_ptr);
            }
        }
        
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
            error!("Could not lock shared state in window procedure");
            return DefWindowProcW(hwnd, msg, wparam, lparam);
        };

        let Some(shared_state) = shared_state_guard.get(&hwnd.0) else {
            // No pipeline for this window, just call default
            return DefWindowProcW(hwnd, msg, wparam, lparam);
        };

        Arc::clone(shared_state)
    };

    // Try to get mutable pipeline data
    let pipeline_data_ptr = {
        PIPELINES.lock().get(&hwnd.0).copied()
    };

    // If we have pipeline data, process message synchronously
    if let Some(data_ptr) = pipeline_data_ptr {
        // Process message with ImGui immediately
        let ctx = &mut (*data_ptr).ctx;
        
        // Update ImGui with this message
        match msg {
            WM_LBUTTONDOWN | WM_LBUTTONUP | WM_RBUTTONDOWN | WM_RBUTTONUP |
            WM_MBUTTONDOWN | WM_MBUTTONUP | WM_MOUSEWHEEL | WM_MOUSEMOVE => {
                // Process mouse message
                imgui_wnd_proc_impl(hwnd, msg, wparam, lparam, ctx);
                
                // Make decision with CURRENT state
                let io = ctx.io();
                if io.want_capture_mouse {
                    // ImGui wants this mouse message - block it
                    return LRESULT(1);
                }
            }
            WM_KEYDOWN | WM_KEYUP | WM_CHAR => {
                // Process keyboard message
                imgui_wnd_proc_impl(hwnd, msg, wparam, lparam, ctx);
                
                // Make decision with CURRENT state
                let io = ctx.io();
                if io.want_capture_keyboard {
                    // ImGui wants this keyboard message - block it
                    return LRESULT(1);
                }
            }
            _ => {
                // Non-input message, just process it
                imgui_wnd_proc_impl(hwnd, msg, wparam, lparam, ctx);
            }
        }
    } else {
        // No pipeline data available (between frames or not initialized yet)
        // Take conservative approach: block all input to prevent issues
        match msg {
            WM_LBUTTONDOWN | WM_LBUTTONUP | WM_RBUTTONDOWN | WM_RBUTTONUP |
            WM_MBUTTONDOWN | WM_MBUTTONUP | WM_MOUSEWHEEL | WM_MOUSEMOVE |
            WM_KEYDOWN | WM_KEYUP | WM_CHAR => {
                // Block input when we can't check ImGui state
                // Better to miss game input than break the overlay
                return LRESULT(1);
            }
            _ => {}
        }
    }

    // Message not blocked - forward to original window procedure
    CallWindowProcW(Some(shared_state.wnd_proc), hwnd, msg, wparam, lparam)
}

// Simplified version of imgui_wnd_proc_impl that works with just Context
unsafe fn imgui_wnd_proc_impl(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    ctx: &mut Context,
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
        WM_KEYDOWN => {
            let vk = wparam.0 as i32;
            if vk < 256 {
                io.add_key_event(vk_to_imgui_key(vk), true);
            }
        }
        WM_KEYUP => {
            let vk = wparam.0 as i32;
            if vk < 256 {
                io.add_key_event(vk_to_imgui_key(vk), false);
            }
        }
        WM_CHAR => {
            if wparam.0 > 0 && wparam.0 < 0x10000 {
                io.add_input_character(wparam.0 as u16 as char);
            }
        }
        _ => {}
    }
}

// Helper to convert VK codes to ImGui keys
fn vk_to_imgui_key(vk: i32) -> imgui::Key {
    use imgui::Key;
    match vk {
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
        0x11 => Key::LeftCtrl,
        0xA2 => Key::LeftCtrl,
        0xA3 => Key::RightCtrl,
        0x10 => Key::LeftShift,
        0xA0 => Key::LeftShift,
        0xA1 => Key::RightShift,
        0x12 => Key::LeftAlt,
        0xA4 => Key::LeftAlt,
        0xA5 => Key::RightAlt,
        0x5B => Key::LeftSuper,
        0x5C => Key::RightSuper,
        0x41 => Key::A,
        0x43 => Key::C,
        0x56 => Key::V,
        0x58 => Key::X,
        0x59 => Key::Y,
        0x5A => Key::Z,
        _ => Key::None,
    }
}
