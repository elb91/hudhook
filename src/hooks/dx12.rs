//! DirectX 12 hooks 

use std::ffi::c_void;
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use imgui::Context;
use tracing::{debug, error, trace, warn};
use windows::core::{Error, Interface, Result, HRESULT};
use windows::Win32::Foundation::BOOL;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12CreateDevice, ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Resource,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_MODE_DESC, DXGI_MODE_SCALING_UNSPECIFIED,
    DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED, DXGI_RATIONAL, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, IDXGIFactory2, IDXGISwapChain, IDXGISwapChain3, DXGI_SWAP_CHAIN_DESC,
    DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH, DXGI_SWAP_EFFECT_FLIP_DISCARD,
    DXGI_USAGE_RENDER_TARGET_OUTPUT,
};

use super::DummyHwnd;
use crate::mh::MhHook;
use crate::renderer::{D3D12RenderEngine, Pipeline};
use crate::{perform_eject, util, Hooks, ImguiRenderLoop, EJECT_REQUESTED, HOOK_EJECTION_BARRIER};

type DXGISwapChainPresentType =
    unsafe extern "system" fn(this: IDXGISwapChain3, sync_interval: u32, flags: u32) -> HRESULT;

type DXGISwapChainResizeBuffersType = unsafe extern "system" fn(
    this: IDXGISwapChain3,
    buffer_count: u32,
    width: u32,
    height: u32,
    new_format: DXGI_FORMAT,
    flags: u32,
) -> HRESULT;

type D3D12CommandQueueExecuteCommandListsType = unsafe extern "system" fn(
    this: ID3D12CommandQueue,
    num_command_lists: u32,
    command_lists: *mut ID3D12CommandList,
);

struct Trampolines {
    dxgi_swap_chain_present: DXGISwapChainPresentType,
    dxgi_swap_chain_resize_buffers: DXGISwapChainResizeBuffersType,
    d3d12_command_queue_execute_command_lists: D3D12CommandQueueExecuteCommandListsType,
}

static TRAMPOLINES: OnceLock<Trampolines> = OnceLock::new();

#[derive(Clone)]
struct SendPtr(*const c_void);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

struct RenderState {
    command_queue: Option<ID3D12CommandQueue>,
    swap_chain: Option<IDXGISwapChain3>,
    pipeline: Option<Pipeline<D3D12RenderEngine>>,
    initialized: bool,
    initializing: bool,
    queue_discovery_complete: bool,
    first_queue_ptr: SendPtr,
}

// Manual Send/Sync implementation - we ensure thread safety through Mutex
unsafe impl Send for RenderState {}
unsafe impl Sync for RenderState {}

impl RenderState {
    fn new() -> Self {
        Self {
            command_queue: None,
            swap_chain: None,
            pipeline: None,
            initialized: false,
            initializing: false,
            queue_discovery_complete: false,
            first_queue_ptr: SendPtr(std::ptr::null()),
        }
    }

    fn reset(&mut self) {
        debug!("Resetting render state");
        
        if let Some(mut pipeline) = self.pipeline.take() {
            // CRITICAL: Extract render loop before cleanup so we can reuse it
            unsafe {
                if let Some(render_loop) = pipeline.take_render_loop() {
                    debug!("Extracted render loop from pipeline before cleanup");
                    RENDER_LOOP.get_or_init(|| render_loop);
                }
            }
            
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipeline.cleanup();
            }));
        }
        
        // IMPORTANT: Keep swap_chain and command_queue - only reset pipeline
        // self.swap_chain = None;  // DON'T RESET THIS
        // self.command_queue = None;  // DON'T RESET THIS
        self.initialized = false;
        self.initializing = false;
    }
}

static RENDER_STATE: OnceLock<Arc<Mutex<RenderState>>> = OnceLock::new();
static mut RENDER_LOOP: OnceLock<Box<dyn ImguiRenderLoop + Send + Sync>> = OnceLock::new();
static RENDERING: AtomicBool = AtomicBool::new(false);
static SAME_QUEUE_COUNT: AtomicU32 = AtomicU32::new(0);
static SKIP_FRAMES: AtomicU32 = AtomicU32::new(60);

// Optional callback that the application can set to detect transitions
static mut TRANSITION_DETECTOR: Option<fn() -> bool> = None;

/// Allows the application to provide a transition detection callback
/// Call this from your main app initialization: 
/// `hudhook::hooks::dx12::set_transition_detector(your_app::is_game_transitioning);`
pub fn set_transition_detector(detector: fn() -> bool) {
    unsafe {
        TRANSITION_DETECTOR = Some(detector);
    }
}

fn check_game_transition() -> bool {
    unsafe {
        if let Some(detector) = TRANSITION_DETECTOR {
            detector()
        } else {
            false
        }
    }
}

fn get_render_state() -> Arc<Mutex<RenderState>> {
    RENDER_STATE
        .get_or_init(|| Arc::new(Mutex::new(RenderState::new())))
        .clone()
}

unsafe fn init_pipeline(
    swap_chain: &IDXGISwapChain3,
    command_queue: &ID3D12CommandQueue,
) -> Result<Pipeline<D3D12RenderEngine>> {
    debug!("Initializing rendering pipeline");
    
    let hwnd = util::try_out_param(|v| swap_chain.GetDesc(v))
        .map(|desc| desc.OutputWindow)?;

    let mut ctx = Context::create();
    ctx.set_ini_filename(None);
    
    let engine = D3D12RenderEngine::new(command_queue, &mut ctx)?;

    let Some(render_loop) = RENDER_LOOP.take() else {
        error!("Render loop not initialized");
        return Err(Error::from_hresult(HRESULT(-1)));
    };

    // Create pipeline - if it fails, restore the render loop
    let pipeline_result = Pipeline::new(hwnd, ctx, engine, render_loop);
    
    match pipeline_result {
        Ok(pipeline) => {
            debug!("Pipeline initialized successfully");
            Ok(pipeline)
        }
        Err((e, render_loop)) => {
            // Restore render loop so we can retry later
            RENDER_LOOP.get_or_init(move || render_loop);
            error!("Failed to create pipeline: {:?}", e);
            Err(e)
        }
    }
}

fn render(swap_chain: &IDXGISwapChain3) -> Result<()> {
    // Check for game transition - if detected, reset and skip
    if check_game_transition() {
        trace!("Game transition active - skipping render");
        let state_lock = get_render_state();
        let mut state = state_lock.lock().unwrap_or_else(|e| e.into_inner());
        if state.initialized {
            state.reset();
        }
        return Ok(());
    }
    
    let state_lock = get_render_state();
    
    // Check if we should initialize (without holding lock for long)
    let should_init = {
        let mut state = state_lock.lock().unwrap_or_else(|e| {
            error!("Mutex poisoned during init check, recovering");
            e.into_inner()
        });
        
        // Update swap chain if needed
        if state.swap_chain.is_none() {
            state.swap_chain = Some(swap_chain.clone());
        }
        
        // Check if we should start initialization
        if !state.initialized && !state.initializing 
            && state.swap_chain.is_some() 
            && state.command_queue.is_some() {
            state.initializing = true;
            true
        } else {
            false
        }
    };
    
    if should_init {
        debug!("Both swap chain and command queue available - initializing");
        
        // Get resources without holding the lock
        let (sc, cq) = {
            let state = state_lock.lock().unwrap_or_else(|e| e.into_inner());
            (state.swap_chain.clone().unwrap(), state.command_queue.clone().unwrap())
        };
        
        // Initialize pipeline WITHOUT holding lock (prevents deadlock)
        let init_result = unsafe { init_pipeline(&sc, &cq) };
        
        // Store result while holding lock
        match init_result {
            Ok(pipeline) => {
                let mut state = state_lock.lock().unwrap_or_else(|e| e.into_inner());
                state.pipeline = Some(pipeline);
                state.initialized = true;
                state.initializing = false;
                debug!("Pipeline ready for rendering");
            }
            Err(e) => {
                error!("Failed to initialize pipeline: {:?}", e);
                let mut state = state_lock.lock().unwrap_or_else(|e| e.into_inner());
                state.initializing = false;
                return Err(e);
            }
        }
    }
    
    // Check if we're ready to render
    let can_render = {
        let state = state_lock.lock().unwrap_or_else(|e| e.into_inner());
        state.initialized && !state.initializing
    };
    
    if !can_render {
        trace!("Waiting for initialization");
        return Ok(());
    }
    
    // Get raw pointer to pipeline while holding lock
    let pipeline_ptr = {
        let mut state = state_lock.lock().unwrap_or_else(|e| e.into_inner());
        state.pipeline.as_mut().map(|p| p as *mut Pipeline<D3D12RenderEngine>)
    };
    
    // Render using the raw pointer
    // SAFETY: We hold RENDERING flag, preventing ResizeBuffers from destroying pipeline
    // The pipeline remains valid because reset() is only called when RENDERING is false
    if let Some(pipeline_ptr) = pipeline_ptr {
        let pipeline = unsafe { &mut *pipeline_ptr };
        
        pipeline.prepare_render()?;
        
        let target: ID3D12Resource = unsafe {
            swap_chain.GetBuffer(swap_chain.GetCurrentBackBufferIndex())?
        };
        
        pipeline.render(target)?;
    }
    
    Ok(())
}

unsafe extern "system" fn dxgi_swap_chain_present_impl(
    swap_chain: IDXGISwapChain3,
    sync_interval: u32,
    flags: u32,
) -> HRESULT {
    let _hook_guard = HOOK_EJECTION_BARRIER.acquire_ejection_guard();
    
    // Skip frames if needed (using atomic operations)
    let skip = SKIP_FRAMES.load(Ordering::Acquire);
    if skip > 0 {
        SKIP_FRAMES.fetch_sub(1, Ordering::Release);
        trace!("Skipping frame: {} remaining", skip - 1);
        
        let Trampolines { dxgi_swap_chain_present, .. } =
            TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");
        return dxgi_swap_chain_present(swap_chain, sync_interval, flags);
    }

    let Trampolines { dxgi_swap_chain_present, .. } =
        TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");

    // Try to acquire render lock (only one render at a time)
    if RENDERING.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
        // Get current command queue to check validity
        let command_queue = {
            let state_lock = get_render_state();
            let state = state_lock.lock().unwrap_or_else(|e| e.into_inner());
            state.command_queue.clone()
        };
        
        if let Some(cq) = command_queue {
            let render_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                render(&swap_chain)
            }));
            
            match render_result {
                Ok(Ok(())) => {
                    trace!("Frame rendered successfully");
                },
                Ok(Err(e)) => {
                    error!("Render error: {e:?}");
                    util::print_dxgi_debug_messages();
                },
                Err(_) => {
                    error!("Render panicked - skipping 60 frames and resetting state");
                    SKIP_FRAMES.store(60, Ordering::Release);
                    
                    // Always reset state on panic, even if mutex is poisoned
                    let state_lock = get_render_state();
                    let mut state = state_lock.lock().unwrap_or_else(|e| {
                        error!("Mutex poisoned during panic recovery");
                        e.into_inner()
                    });
                    state.reset();
                }
            }
        }
        
        RENDERING.store(false, Ordering::Release);
    }

    trace!("Calling original Present");
    let result = dxgi_swap_chain_present(swap_chain, sync_interval, flags);

    if EJECT_REQUESTED.load(Ordering::SeqCst) {
        perform_eject();
    }

    result
}

unsafe extern "system" fn dxgi_swap_chain_resize_buffers_impl(
    swap_chain: IDXGISwapChain3,
    buffer_count: u32,
    width: u32,
    height: u32,
    new_format: DXGI_FORMAT,
    flags: u32,
) -> HRESULT {
    let _hook_guard = HOOK_EJECTION_BARRIER.acquire_ejection_guard();
    
    debug!(
        "ResizeBuffers called: {}x{}, buffers={}, format={:?}",
        width, height, buffer_count, new_format
    );
    
    // Wait for any active rendering to complete (with timeout)
    let mut wait_count = 0;
    while RENDERING.load(Ordering::Acquire) && wait_count < 1000 {
        std::hint::spin_loop();
        wait_count += 1;
    }
    
    if wait_count >= 1000 {
        error!("Timeout waiting for render completion");
    }
    
    // Now safe to reset state
    let state_lock = get_render_state();
    let mut state = state_lock.lock().unwrap_or_else(|e| {
        error!("Mutex poisoned in ResizeBuffers, recovering");
        e.into_inner()
    });
    state.reset();
    drop(state);
    
    let Trampolines { dxgi_swap_chain_resize_buffers, .. } =
        TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");

    debug!("Calling original ResizeBuffers");
    let result = dxgi_swap_chain_resize_buffers(
        swap_chain, 
        buffer_count, 
        width, 
        height, 
        new_format, 
        flags
    );
    
    if result.is_err() {
        error!("ResizeBuffers failed: {:?}", result);
    } else {
        debug!("ResizeBuffers succeeded - skipping 30 frames for stability");
        SKIP_FRAMES.store(30, Ordering::Release);
    }
    
    result
}

unsafe extern "system" fn d3d12_command_queue_execute_command_lists_impl(
    command_queue: ID3D12CommandQueue,
    num_command_lists: u32,
    command_lists: *mut ID3D12CommandList,
) {
    let _hook_guard = HOOK_EJECTION_BARRIER.acquire_ejection_guard();
    
    let desc = command_queue.GetDesc();
    
    // Only capture DIRECT queues (Type 0)
    if desc.Type == D3D12_COMMAND_LIST_TYPE_DIRECT {
        let state_lock = get_render_state();
        let mut state = state_lock.lock().unwrap_or_else(|e| {
            error!("Mutex poisoned in ExecuteCommandLists, recovering");
            e.into_inner()
        });
        
        let queue_ptr = command_queue.as_raw() as *const c_void;
        
        // Validate pointer is not null
        if queue_ptr.is_null() {
            error!("Command queue pointer is null");
            drop(state);
            
            let Trampolines { d3d12_command_queue_execute_command_lists, .. } =
                TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");
            d3d12_command_queue_execute_command_lists(command_queue, num_command_lists, command_lists);
            return;
        }
        
        // Queue discovery phase
        if !state.queue_discovery_complete {
            if state.command_queue.is_none() {
                debug!(
                    "First DIRECT queue: Lists={}, Ptr={:p}",
                    num_command_lists, queue_ptr
                );
                state.command_queue = Some(command_queue.clone());
                state.first_queue_ptr = SendPtr(queue_ptr);
            } else {
                let first_ptr = state.first_queue_ptr.0;
                
                if first_ptr != queue_ptr {
                    debug!(
                        "Second DIRECT queue found: Lists={}, Ptr={:p} - USING THIS",
                        num_command_lists, queue_ptr
                    );
                    state.command_queue = Some(command_queue.clone());
                    state.queue_discovery_complete = true;
                    
                    if state.initialized {
                        debug!("Resetting pipeline for main rendering queue");
                        let old_sc = state.swap_chain.clone();
                        state.reset();
                        state.swap_chain = old_sc;
                        state.command_queue = Some(command_queue.clone());
                        state.queue_discovery_complete = true;
                    }
                } else {
                    let count = SAME_QUEUE_COUNT.fetch_add(1, Ordering::Relaxed);
                    
                    if count > 100 {
                        debug!("Only one DIRECT queue exists - using it");
                        state.queue_discovery_complete = true;
                    }
                }
            }
        }
        
        drop(state);
    }

    let Trampolines { d3d12_command_queue_execute_command_lists, .. } =
        TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");

    d3d12_command_queue_execute_command_lists(command_queue, num_command_lists, command_lists);
}

fn get_target_addrs() -> (
    DXGISwapChainPresentType,
    DXGISwapChainResizeBuffersType,
    D3D12CommandQueueExecuteCommandListsType,
) {
    debug!("Getting DirectX 12 vtable addresses");
    
    let dummy_hwnd = DummyHwnd::new();

    let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(0) }
        .expect("Failed to create DXGI factory");
    
    let adapter = unsafe { factory.EnumAdapters(0) }
        .expect("Failed to enumerate adapters");

    let device: ID3D12Device = util::try_out_ptr(|v| unsafe {
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, v)
    })
    .expect("D3D12CreateDevice failed");

    let command_queue: ID3D12CommandQueue = unsafe {
        device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: 0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        })
    }
    .expect("Failed to create command queue");

    let swap_chain: IDXGISwapChain = util::try_out_ptr(|v| unsafe {
        factory.CreateSwapChain(
            &command_queue,
            &DXGI_SWAP_CHAIN_DESC {
                BufferDesc: DXGI_MODE_DESC {
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    ScanlineOrdering: DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED,
                    Scaling: DXGI_MODE_SCALING_UNSPECIFIED,
                    Width: 640,
                    Height: 480,
                    RefreshRate: DXGI_RATIONAL {
                        Numerator: 60,
                        Denominator: 1,
                    },
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: 2,
                OutputWindow: dummy_hwnd.hwnd(),
                Windowed: BOOL(1),
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH.0 as _,
            },
            v,
        )
        .ok()
    })
    .expect("Failed to create swap chain");

    unsafe {
        let sc_vtable = *(swap_chain.as_raw() as *const *const usize);
        let q_vtable = *(command_queue.as_raw() as *const *const usize);
        
        let present_ptr = *sc_vtable.add(8);
        let resize_buffers_ptr = *sc_vtable.add(13);
        let execute_command_lists_ptr = *q_vtable.add(10);
        
        debug!("Vtable addresses extracted:");
        debug!("  Present:              {:p}", present_ptr as *const c_void);
        debug!("  ResizeBuffers:        {:p}", resize_buffers_ptr as *const c_void);
        debug!("  ExecuteCommandLists:  {:p}", execute_command_lists_ptr as *const c_void);
        
        (
            mem::transmute(present_ptr),
            mem::transmute(resize_buffers_ptr),
            mem::transmute(execute_command_lists_ptr),
        )
    }
}

pub struct ImguiDx12Hooks([MhHook; 3]);

impl ImguiDx12Hooks {
    pub unsafe fn new<T>(t: T) -> Self
    where
        T: ImguiRenderLoop + Send + Sync + 'static,
    {
        debug!("Installing DirectX 12 hooks");
        
        let (present_addr, resize_addr, execute_addr) = get_target_addrs();

        let hook_present = MhHook::new(
            present_addr as *mut _,
            dxgi_swap_chain_present_impl as *mut _,
        )
        .expect("Failed to create Present hook");
        
        let hook_resize = MhHook::new(
            resize_addr as *mut _,
            dxgi_swap_chain_resize_buffers_impl as *mut _,
        )
        .expect("Failed to create ResizeBuffers hook");
        
        let hook_execute = MhHook::new(
            execute_addr as *mut _,
            d3d12_command_queue_execute_command_lists_impl as *mut _,
        )
        .expect("Failed to create ExecuteCommandLists hook");

        RENDER_LOOP.get_or_init(|| Box::new(t));

        TRAMPOLINES.get_or_init(|| Trampolines {
            dxgi_swap_chain_present: mem::transmute(hook_present.trampoline()),
            dxgi_swap_chain_resize_buffers: mem::transmute(hook_resize.trampoline()),
            d3d12_command_queue_execute_command_lists: mem::transmute(hook_execute.trampoline()),
        });

        debug!("DirectX 12 hooks installed successfully");
        Self([hook_present, hook_resize, hook_execute])
    }
}

impl Hooks for ImguiDx12Hooks {
    fn from_render_loop<T>(t: T) -> Box<Self>
    where
        Self: Sized,
        T: ImguiRenderLoop + Send + Sync + 'static,
    {
        Box::new(unsafe { Self::new(t) })
    }

    fn hooks(&self) -> &[MhHook] {
        &self.0
    }

    unsafe fn unhook(&mut self) {
        debug!("Unhooking DirectX 12");
        
        // Wait for rendering to complete
        let mut wait_count = 0;
        while RENDERING.load(Ordering::Acquire) && wait_count < 1000 {
            std::hint::spin_loop();
            wait_count += 1;
        }
        
        let state_lock = get_render_state();
        let mut state = state_lock.lock().unwrap_or_else(|e| {
            error!("Mutex poisoned during unhook, recovering");
            e.into_inner()
        });
        state.reset();
        drop(state);
        
        RENDERING.store(false, Ordering::Release);
        
        debug!("DirectX 12 unhook complete");
    }
}
