let device, context, format, queue;
let renderPipeline;
let imageUploaded = false;

// Basic variables to mock 3D transform matrices (for interior camera view space)
let parameters = {
    brightness: 1.0,
    realism: 1.2,
    roughness: 0.5,
    lightIntensity: 10.0
};

// UI Events Setup
document.getElementById('imageInput').addEventListener('change', handleImageUpload);
document.getElementById('brightness').addEventListener('input', (e) => parameters.brightness = parseFloat(e.target.value));
document.getElementById('realism').addEventListener('input', (e) => parameters.realism = parseFloat(e.target.value));
document.getElementById('roughness').addEventListener('input', (e) => parameters.roughness = parseFloat(e.target.value));
document.getElementById('lightIntensity').addEventListener('input', (e) => parameters.lightIntensity = parseFloat(e.target.value));

function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Display image thumbnail
    const reader = new FileReader();
    reader.onload = function(event) {
        document.getElementById('upload-text').style.display = 'none';
        const thumbContainer = document.getElementById('thumbContainer');
        const thumbImg = document.getElementById('thumbImg');
        thumbImg.src = event.target.result;
        thumbContainer.style.display = 'block';

        // Trigger ML generation simulation
        startInteriorProcessing();
    }
    reader.readAsDataURL(file);
}

function startInteriorProcessing() {
    const progressSection = document.getElementById('progressSection');
    const progressFill = document.getElementById('progressFill');
    progressSection.style.display = 'block';
    progressFill.style.width = '0%';

    let progress = 0;
    const interval = setInterval(() => {
        progress += 4;
        progressFill.style.width = progress + '%';
        if (progress >= 100) {
            clearInterval(interval);
            setTimeout(() => {
                progressSection.style.display = 'none';
                imageUploaded = true; // Signals render loop to output path-traced scene representation instead of flat canvas
            }, 500);
        }
    }, 100);
}

// -------------------------
// WEBGPU RENDER ENGINE
// -------------------------

async function initWebGPU() {
    if (!navigator.gpu) {
        throw new Error("WebGPU is not supported on this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No appropriate GPU adapter found.");

    device = await adapter.requestDevice();
    queue = device.queue;

    const canvas = document.getElementById("engineCanvas");
    
    // Auto-resize viewport canvas
    canvas.width = canvas.parentElement.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.parentElement.clientHeight * window.devicePixelRatio;

    context = canvas.getContext("webgpu");
    format = navigator.gpu.getPreferredCanvasFormat();

    context.configure({
        device: device,
        format: format,
        alphaMode: "premultiplied",
    });
}

// The core shader logic mapping the inputs & textures
const shaders = `
struct FragmentInput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex : u32) -> FragmentInput {
    var pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>( 1.0, -1.0)
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0)
    );

    var output: FragmentInput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.uv = uvs[vertexIndex];
    return output;
}

struct Params {
    brightness: f32,
    realism: f32,
    roughness: f32,
    lightIntensity: f32,
    imageUploaded: f32,
}
@group(0) @binding(0) var<uniform> params : Params;

@fragment
fn fragmentMain(input : FragmentInput) -> @location(0) vec4<f32> {
    if (params.imageUploaded < 0.5) {
        // Fallback grid when idle
        let grid = sin(input.uv.x * 40.0) * sin(input.uv.y * 30.0);
        let color = select(vec3<f32>(0.08), vec3<f32>(0.12), grid > 0.0);
        return vec4<f32>(color, 1.0);
    }
    
    // Simulate a complex path-traced scene evaluation logic using parameters 
    var col = vec3<f32>(0.4, 0.35, 0.3) * params.brightness;
    
    // Artificial interior geometric/lighting gradient to simulate ray-marching/re-lighting
    let centerDist = distance(input.uv, vec2<f32>(0.5, 0.5));
    let ambientLight = (1.0 - centerDist) * params.lightIntensity * 0.15;
    
    col = col * (ambientLight + params.realism * 0.5);
    
    // Specular calculation highlights (simulating physical reflection of room spotlights)
    let spec = pow(max(0.0, 1.0 - centerDist - params.roughness), 8.0) * params.lightIntensity * 0.3;
    col += vec3<f32>(spec);

    return vec4<f32>(col, 1.0);
}
`;

async function startEngine() {
    await initWebGPU();

    const shaderModule = device.createShaderModule({ code: shaders });

    // Setup parameter uniform buffer to interact with shader
    const parameterBufferSize = 32; // 32 bytes to pack floats nicely
    const parameterBuffer = device.createBuffer({
        size: parameterBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUSHaderStage.FRAGMENT,
            buffer: { type: "uniform" }
        }]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: parameterBuffer }
        }]
    });

    renderPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        vertex: { module: shaderModule, entryPoint: "vertexMain" },
        fragment: {
            module: shaderModule,
            entryPoint: "fragmentMain",
            targets: [{ format: format }],
        },
        primitive: { topology: "triangle-strip" },
    });

    function renderFrame() {
        // Update Uniform data directly every frame
        const data = new Float32Array([
            parameters.brightness,
            parameters.realism,
            parameters.roughness,
            parameters.lightIntensity,
            imageUploaded ? 1.0 : 0.0,
            0.0, 0.0, 0.0 // padding bytes to meet 32 byte spacing
        ]);
        queue.writeBuffer(parameterBuffer, 0, data);

        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

        const renderPassDescriptor = {
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                loadOp: "clear",
                storeOp: "store",
            }],
        };

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(renderPipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.draw(4); 
        passEncoder.end();

        queue.submit([commandEncoder.finish()]);
        
        requestAnimationFrame(renderFrame);
    }

    requestAnimationFrame(renderFrame);
}

startEngine().catch(console.error);
