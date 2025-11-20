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

static mut TRAMPOLINES: OnceLock<Trampolines> = OnceLock::new();

/// Wrapper to make raw pointer Send (we ensure thread safety manually)
struct SendPtr(*const c_void);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Global render state
struct RenderState {
    swap_chain: Option<IDXGISwapChain3>,
    command_queue: Option<ID3D12CommandQueue>,
    pipeline: Option<Pipeline<D3D12RenderEngine>>,
    initialized: bool,
    // Queue discovery tracking
    queue_discovery_complete: bool,
    first_queue_ptr: SendPtr,
}

// We manually ensure thread safety through Mutex and careful access patterns
unsafe impl Send for RenderState {}
unsafe impl Sync for RenderState {}

impl RenderState {
    fn new() -> Self {
        Self {
            swap_chain: None,
            command_queue: None,
            pipeline: None,
            initialized: false,
            queue_discovery_complete: false,
            first_queue_ptr: SendPtr(std::ptr::null()),
        }
    }
    
    /// Reset state but preserve queue discovery info
    fn reset(&mut self) {
        debug!("RenderState::reset - cleaning up resources");
        
        if let Some(mut pipeline) = self.pipeline.take() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipeline.cleanup();
            }));
        }
        
        self.swap_chain = None;
        self.command_queue = None;
        self.initialized = false;
        // Don't reset queue_discovery_complete - we already know which queue to use
        
        debug!("RenderState reset complete");
    }
}

static RENDER_STATE: OnceLock<Arc<Mutex<RenderState>>> = OnceLock::new();

fn get_render_state() -> Arc<Mutex<RenderState>> {
    RENDER_STATE
        .get_or_init(|| Arc::new(Mutex::new(RenderState::new())))
        .clone()
}

static mut RENDER_LOOP: OnceLock<Box<dyn ImguiRenderLoop + Send + Sync>> = OnceLock::new();
static RENDERING: AtomicBool = AtomicBool::new(false);
static SAME_QUEUE_COUNT: AtomicU32 = AtomicU32::new(0);

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

    let pipeline = Pipeline::new(hwnd, ctx, engine, render_loop).map_err(|(e, render_loop)| {
        RENDER_LOOP.get_or_init(move || render_loop);
        e
    })?;

    debug!("Pipeline initialized successfully");
    Ok(pipeline)
}

fn render(swap_chain: &IDXGISwapChain3) -> Result<()> {
    let state_lock = get_render_state();
    let mut state = state_lock.lock().unwrap();
    
    if !state.initialized {
        if state.swap_chain.is_none() {
            state.swap_chain = Some(swap_chain.clone());
        }
        
        if let (Some(sc), Some(cq)) = (&state.swap_chain, &state.command_queue) {
            debug!("Both swap chain and command queue available - initializing");
            
            match unsafe { init_pipeline(sc, cq) } {
                Ok(pipeline) => {
                    state.pipeline = Some(pipeline);
                    state.initialized = true;
                    debug!("Pipeline ready for rendering");
                }
                Err(e) => {
                    error!("Failed to initialize pipeline: {:?}", e);
                    return Err(e);
                }
            }
        } else {
            trace!("Waiting for command queue before initialization");
            return Ok(());
        }
    }
    
    if let Some(pipeline) = &mut state.pipeline {
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
    
    static mut SKIP_FRAMES: u32 = 60;
    static mut LAST_SWAP_CHAIN_PTR: *const c_void = std::ptr::null();
    
    let current_ptr = swap_chain.as_raw() as *const c_void;
    
    // Detect swap chain changes
    if current_ptr != LAST_SWAP_CHAIN_PTR {
        if !LAST_SWAP_CHAIN_PTR.is_null() {
            error!("Swap chain changed: {:p} -> {:p}", LAST_SWAP_CHAIN_PTR, current_ptr);
            
            let state_lock = get_render_state();
            if let Ok(mut state) = state_lock.lock() {
                state.reset();
                state.swap_chain = Some(swap_chain.clone());
            }
            
            SKIP_FRAMES = 60;
            RENDERING.store(false, Ordering::Release);
        }
        LAST_SWAP_CHAIN_PTR = current_ptr;
    }
    
    // Skip frames if needed
    if SKIP_FRAMES > 0 {
        SKIP_FRAMES -= 1;
        trace!("Skipping frame: {} remaining", SKIP_FRAMES);
        
        let Trampolines { dxgi_swap_chain_present, .. } =
            TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");
        return dxgi_swap_chain_present(swap_chain, sync_interval, flags);
    }

    let Trampolines { dxgi_swap_chain_present, .. } =
        TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");

    // Try to render (only one render at a time)
    if RENDERING.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
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
                error!("Render panicked - skipping 60 frames");
                SKIP_FRAMES = 60;
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
    
    error!(
        "ResizeBuffers called: {}x{}, buffers={}, format={:?}",
        width, height, buffer_count, new_format
    );
    
    RENDERING.store(false, Ordering::Release);
    
    let state_lock = get_render_state();
    if let Ok(mut state) = state_lock.lock() {
        state.reset();
    }
    
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
        debug!("ResizeBuffers succeeded");
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
        let mut state = match state_lock.lock() {
            Ok(guard) => guard,
            Err(_) => {
                error!("Failed to lock render state");
                let Trampolines { d3d12_command_queue_execute_command_lists, .. } =
                    TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");
                d3d12_command_queue_execute_command_lists(command_queue, num_command_lists, command_lists);
                return;
            }
        };
        
        let queue_ptr = command_queue.as_raw() as *const c_void;
        
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
        
        // Explicitly drop the lock before calling trampoline
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
        
        TRAMPOLINES.take();
        
        let state_lock = get_render_state();
        if let Ok(mut state) = state_lock.lock() {
            state.reset();
        }
        
        RENDER_LOOP.take();
        RENDERING.store(false, Ordering::Release);
        
        debug!("DirectX 12 unhook complete");
    }
}
