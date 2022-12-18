import PicoGL from "../node_modules/picogl/build/module/picogl.js";
import {mat4, vec3, mat3, vec4, vec2} from "../node_modules/gl-matrix/esm/index.js";

import {positions, normals, uvs, indices} from "../blender/DiscoSuzanne.js"
import {positions as planePositions, indices as planeIndices} from "../blender/plane.js";
import {positions as secondPositions, uvs as secondUvs, indices as secondIndices} from "../blender/Ico.js"








//parameters
let baseColor = vec3.fromValues(1, 1, 1);
let ambientLightColor = vec3.fromValues(0.1, 0.1, 1.0);
let numberOfPointLights = 2;
let pointLightColors = [vec3.fromValues(1.0, 1.0, 1.0), vec3.fromValues(0.6, 0.1, 0.2)];
let pointLightInitialPositions = [vec3.fromValues(5, 0, 2), vec3.fromValues(-5, 0, 2)];
let pointLightPositions = [vec3.create(), vec3.create()];





let skyboxPositions = new Float32Array([
    -1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    -1.0, -1.0, 1.0,
    1.0, -1.0, 1.0
]);

let skyboxTriangles = new Uint32Array([
    0, 2, 1,
    2, 3, 1
]);




//POST
let postPositions = new Float32Array([
    0.0, 1.0,
    1.0, 1.0,
    0.0, 0.0,
    1.0, 0.0,
]);

let postIndices = new Uint32Array([
    0, 2, 1,
    2, 3, 1
]);








//Light
let lightCalculationShader = `
    uniform vec3 cameraPosition;
    uniform vec3 baseColor;    

    uniform vec3 ambientLightColor;    
    uniform vec3 lightColors[${numberOfPointLights}];        
    uniform vec3 lightPositions[${numberOfPointLights}];
    
    // This function calculates light reflection using Phong reflection model (ambient + diffuse + specular)
    vec4 calculateLights(vec3 normal, vec3 position) {
        float ambientIntensity = 0.5;
        float diffuseIntensity = 1.0;
        float specularIntensity = 2.0;
        float specularPower = 100.0;
        float metalness = 0.0;

        vec3 viewDirection = normalize(cameraPosition.xyz - position);
        vec3 color = baseColor * ambientLightColor * ambientIntensity;
                
        for (int i = 0; i < lightPositions.length(); i++) {
            vec3 lightDirection = normalize(lightPositions[i] - position);
            
            // Lambertian reflection (ideal diffuse of matte surfaces) is also a part of Phong model                        
            float diffuse = max(dot(lightDirection, normal), 0.0);                                    
            color += baseColor * lightColors[i] * diffuse * diffuseIntensity;
                      
            // Phong specular highlight 
            //float specular = pow(max(dot(viewDirection, reflect(-lightDirection, normal)), 0.0), specularPower);
            
            // Blinn-Phong improved specular highlight
            float specular = pow(max(dot(normalize(lightDirection + viewDirection), normal), 0.0), specularPower);
            color += mix(vec3(1.0), baseColor, metalness) * lightColors[i] * specular * specularIntensity;
        }
        return vec4(color, 1.0);
    }
`;


let fragmentShader = `
    #version 300 es
    precision highp float;
    precision highp sampler2DShadow;

    ${lightCalculationShader}
    
    //uniform samplerCube cubemap;
    uniform sampler2D tex;    
        
    uniform sampler2DShadow shadowMap;
    

    in vec3 vPosition;    
    in vec3 vNormal;
    in vec4 vColor;
    in vec3 viewDir;
    in vec4 vPositionFromLight;
    in vec3 vModelPosition;
    in vec2 vUv;

    
    out vec4 outColor;
    
    void main()
    {    
        //vec3 reflectedDir = reflect(viewDir, normalize(vNormal));
        
        // Try using a higher mipmap LOD to get a rough material effect without any performance impact
        //outColor = textureLod(cubemap, reflectedDir, 10.0);
        outColor = calculateLights(normalize(vNormal), viewDir) * texture(tex, vUv);
        
    }
`;

// language=GLSL
let vertexShader = `
    #version 300 es
            
    uniform mat4 modelViewProjectionMatrix;
    uniform mat4 modelMatrix;
    uniform mat3 normalMatrix;
    uniform vec3 cameraPosition;
    
    layout(location=0) in vec4 position;
    layout(location=1) in vec3 normal;
    layout(location=2) in vec2 uv;
        
    out vec2 vUv;
    out vec3 vPosition;
    out vec3 vNormal;
    out vec3 viewDir;
    //out vec4 vColor;
    
    void main()
    {

        vec4 worldPosition = modelMatrix * position;
        vPosition = worldPosition.xyz;
        vNormal = normalMatrix * normal;

        //vColor = calculateLights(normalize(vNormal), vPosition);

        vec4 posi = position;
        posi.xyz *= 0.5;
        gl_Position = modelViewProjectionMatrix * posi;           
        vUv = uv;
        viewDir = (modelMatrix * position).xyz - cameraPosition;                
        
    }
`;

// language=GLSL
let mirrorFragmentShader = `
    #version 300 es
    precision highp float;
    
    uniform sampler2D reflectionTex;
    uniform sampler2D distortionMap;
    uniform vec2 screenSize;
    
    in vec2 vUv;        
        
    out vec4 outColor;
    
    void main()
    {                        
        vec2 screenPos = gl_FragCoord.xy / screenSize;
        
        // 0.03 is a mirror distortion factor, try making a larger distortion         
        screenPos.x += (texture(distortionMap, vUv).r - 0.5) * 0.2;
        outColor = texture(reflectionTex, screenPos);
    }
`;

// language=GLSL
let mirrorVertexShader = `
    #version 300 es
            
    uniform mat4 modelViewProjectionMatrix;
    
    layout(location=0) in vec4 position;   
    layout(location=1) in vec2 uv;
    
    out vec2 vUv;
        
    void main()
    {
        vUv = uv;
        vec4 pos = position;
        pos.xyz *= 3.0;
        //pos.y -= 3.0;
        gl_Position = modelViewProjectionMatrix * pos;
    }
`;

// Skybox
let skyboxVertexShader = `
    #version 300 es
    
    layout(location=0) in vec4 position;
    out vec4 v_position;
    
    void main() {
        v_position = vec4(position.xz, 1.0, 1.0);
        gl_Position = v_position;
    }
`;

// Skybox
let skyboxFragmentShader = `
    #version 300 es
    precision mediump float;
    
    uniform samplerCube cubemap;
    uniform mat4 viewProjectionInverse;
    
    in vec4 v_position;
    
    out vec4 outColor;
    
    void main() {
      vec4 t = viewProjectionInverse * v_position;
      outColor = texture(cubemap, normalize(t.xyz / t.w));
    }
`;



// language=GLSL
let postFragmentShader = `
    #version 300 es
    precision mediump float;
    
    uniform sampler2D tex;
    uniform sampler2D depthTex;
    uniform float time;
    uniform sampler2D noiseTex;
    
    in vec4 v_position;
    
    out vec4 outColor;
    
    vec4 depthOfField(vec4 col, float depth, vec2 uv) {
        vec4 blur = vec4(0.0);
        float n = 0.0;
        for (float u = -1.0; u <= 1.0; u += 0.4)    
            for (float v = -1.0; v <= 1.0; v += 0.4) {
                float factor = abs(depth - 0.995) * 350.0;
                blur += texture(tex, uv + vec2(u, v) * factor * 0.02);
                n += 1.0;
            }                
        return blur / n;
    }
    
    vec4 ambientOcclusion(vec4 col, float depth, vec2 uv) {
        if (depth == 1.0) return col;
        for (float u = -2.0; u <= 2.0; u += 0.4)    
            for (float v = -2.0; v <= 2.0; v += 0.4) {                
                float d = texture(depthTex, uv + vec2(u, v) * 0.01).r;
                if (d != 1.0) {
                    float diff = abs(depth - d);
                    col *= 1.0 - diff * 30.0;
                }
            }
        return col;        
    }   
    
    float random(vec2 seed) {
        return texture(noiseTex, seed * 5.0 + sin(time * 543.12) * 54.12).r - 0.5;
    }
    
    void main() {
        vec4 col = texture(tex, v_position.xy);
        float depth = texture(depthTex, v_position.xy).r;
        
        // Chromatic aberration 
        //vec2 caOffset = vec2(0.01, 0.0);
        //col.r = texture(tex, v_position.xy - caOffset).r;
        //col.b = texture(tex, v_position.xy + caOffset).b;
        
        // Depth of field
        //col = depthOfField(col, depth, v_position.xy);

        // Noise         
        //col.rgb += (2.0 - col.rgb) * random(v_position.xy) * 0.1;
        
        // Contrast + Brightness
        col = pow(col, vec4(1.8)) * 0.8;
        
        // Color curves
        //col.rgb = col.rgb * vec3(1.2, 1.1, 1.0) + vec3(0.0, 0.05, 0.2);
        
        // Ambient Occlusion
        //col = ambientOcclusion(col, depth, v_position.xy);                
        
        // Invert
        //col.rgb = 1.0 - col.rgb;
        
        // Fog
        //col.rgb = col.rgb + vec3((depth - 0.992) * 200.0);         
                        
        outColor = col;
    }
`;

// language=GLSL
let postVertexShader = `
    #version 300 es
    
    layout(location=0) in vec4 position;
    out vec4 v_position;
    
    void main() {
        v_position = position;
        gl_Position = position * 2.0 - 1.0;
    }
`;









let program = app.createProgram(vertexShader, fragmentShader);
let skyboxProgram = app.createProgram(skyboxVertexShader, skyboxFragmentShader);
let mirrorProgram = app.createProgram(mirrorVertexShader, mirrorFragmentShader);
let postProgram = app.createProgram(postVertexShader.trim(), postFragmentShader.trim());




let vertexArray = app.createVertexArray()
    .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 3, positions))
    .vertexAttributeBuffer(1, app.createVertexBuffer(PicoGL.FLOAT, 3, normals))
    .vertexAttributeBuffer(2, app.createVertexBuffer(PicoGL.FLOAT, 2, uvs))
    .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, indices));
    





const secondPositionsBuffer = app.createVertexBuffer(PicoGL.FLOAT, 3, secondPositions);
const secondUvsBuffer = app.createVertexBuffer(PicoGL.FLOAT, 2, secondUvs);
const secondIndicesBuffer = app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, secondIndices);






let skyboxArray = app.createVertexArray()
    .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 3, planePositions))
    .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, planeIndices));
/*
let skyboxArray = app.createVertexArray()
    .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 3, skyboxPositions))
    .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, skyboxTriangles));*/

let mirrorArray = app.createVertexArray()
    .vertexAttributeBuffer(0, secondPositionsBuffer)
    .vertexAttributeBuffer(1, secondUvsBuffer)
    .indexBuffer(secondIndicesBuffer);

//postproc
let postArray = app.createVertexArray()
    .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 2, postPositions))
    .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, postIndices));
        





//??????
let colorTarget = app.createTexture2D(app.width, app.height, {magFilter: PicoGL.LINEAR, wrapS: PicoGL.CLAMP_TO_EDGE, wrapR: PicoGL.CLAMP_TO_EDGE});
let depthTarget = app.createTexture2D(app.width, app.height, {internalFormat: PicoGL.DEPTH_COMPONENT32F, type: PicoGL.FLOAT});
let buffer = app.createFramebuffer().colorTarget(0, colorTarget).depthTarget(depthTarget);






// Change the reflection texture resolution to checkout the difference
let reflectionResolutionFactor = 0.6;
let reflectionColorTarget = app.createTexture2D(app.width * reflectionResolutionFactor, app.height * reflectionResolutionFactor, {magFilter: PicoGL.LINEAR});
let reflectionDepthTarget = app.createTexture2D(app.width * reflectionResolutionFactor, app.height * reflectionResolutionFactor, {internalFormat: PicoGL.DEPTH_COMPONENT16});
let reflectionBuffer = app.createFramebuffer().colorTarget(0, reflectionColorTarget).depthTarget(reflectionDepthTarget);

let projMatrix = mat4.create();
let viewMatrix = mat4.create();
let viewProjMatrix = mat4.create();
let modelMatrix = mat4.create();
let modelViewMatrix = mat4.create();
let modelViewProjectionMatrix = mat4.create();
let rotateXMatrix = mat4.create();
let rotateYMatrix = mat4.create();
let mirrorModelMatrix = mat4.create();
let mirrorModelViewProjectionMatrix = mat4.create();
let skyboxViewProjectionInverse = mat4.create();
let cameraPosition = vec3.create();










function calculateSurfaceReflectionMatrix(reflectionMat, mirrorModelMatrix, surfaceNormal) {
    let normal = vec3.transformMat3(vec3.create(), surfaceNormal, mat3.normalFromMat4(mat3.create(), mirrorModelMatrix));
    let pos = mat4.getTranslation(vec3.create(), mirrorModelMatrix);
    let d = -vec3.dot(normal, pos);
    let plane = vec4.fromValues(normal[0], normal[1], normal[2], d);

    reflectionMat[0] = (1 - 2 * plane[0] * plane[0]);
    reflectionMat[4] = ( - 2 * plane[0] * plane[1]);
    reflectionMat[8] = ( - 2 * plane[0] * plane[2]);
    reflectionMat[12] = ( - 2 * plane[3] * plane[0]);

    reflectionMat[1] = ( - 2 * plane[1] * plane[0]);
    reflectionMat[5] = (1 - 2 * plane[1] * plane[1]);
    reflectionMat[9] = ( - 2 * plane[1] * plane[2]);
    reflectionMat[13] = ( - 2 * plane[3] * plane[1]);

    reflectionMat[2] = ( - 2 * plane[2] * plane[0]);
    reflectionMat[6] = ( - 2 * plane[2] * plane[1]);
    reflectionMat[10] = (1 - 2 * plane[2] * plane[2]);
    reflectionMat[14] = ( - 2 * plane[3] * plane[2]);

    reflectionMat[3] = 0;
    reflectionMat[7] = 0;
    reflectionMat[11] = 0;
    reflectionMat[15] = 1;

    return reflectionMat;
}







//calling out files
async function loadTexture(fileName) {
    return await createImageBitmap(await (await fetch("images/" + fileName)).blob());
}

//cubemap image files
const cubemap = app.createCubemap({
    negX: await loadTexture("nx.png"),
    posX: await loadTexture("px.png"),
    negY: await loadTexture("ny.png"),
    posY: await loadTexture("py.png"),
    negZ: await loadTexture("nz.png"),
    posZ: await loadTexture("pz.png")
});









//calling noise file for Ico 
const tex = await loadTexture("noise.png");
let mirrorDrawCall = app.createDrawCall(mirrorProgram, mirrorArray)
    .texture("reflectionTex", reflectionColorTarget)
    .texture("distortionMap", app.createTexture2D(tex, tex.width, tex.height, ));


    /*let drawCall = app.createDrawCall(program, vertexArray)
    .texture("cubemap", cubemap)
    .uniform("baseColor", baseColor)
    .uniform("ambientLightColor", ambientLightColor);*/

// cant use tex cause fragment shader dont like the stuff I try to put in.
let drawCall = app.createDrawCall(program, vertexArray)
    .texture("tex", app.createTexture2D(await loadTexture("texture.png")))
    .uniform("ambientLightColor", ambientLightColor);


let skyboxDrawCall = app.createDrawCall(skyboxProgram, skyboxArray)
    .texture("cubemap", cubemap);










const positionsBuffer = new Float32Array(numberOfPointLights * 3);
const colorsBuffer = new Float32Array(numberOfPointLights * 3);







function renderReflectionTexture()
{
    app.drawFramebuffer(reflectionBuffer);
    app.viewport(0, 0, reflectionColorTarget.width, reflectionColorTarget.height);

    app.gl.cullFace(app.gl.FRONT);

    let reflectionMatrix = calculateSurfaceReflectionMatrix(mat4.create(), mirrorModelMatrix, vec3.fromValues(0, 1, 0));
    let reflectionViewMatrix = mat4.mul(mat4.create(), viewMatrix, reflectionMatrix);
    let reflectionCameraPosition = vec3.transformMat4(vec3.create(), cameraPosition, reflectionMatrix);
    drawObjects(reflectionCameraPosition, reflectionViewMatrix);

    app.gl.cullFace(app.gl.BACK);
    app.defaultDrawFramebuffer();
    app.defaultViewport();
}










function drawObjects(cameraPosition, viewMatrix) {
    mat4.multiply(viewProjMatrix, projMatrix, viewMatrix);

    mat4.multiply(modelViewMatrix, viewMatrix, modelMatrix);
    mat4.multiply(modelViewProjectionMatrix, viewProjMatrix, modelMatrix);

    let skyboxViewProjectionMatrix = mat4.create();
    mat4.mul(skyboxViewProjectionMatrix, projMatrix, viewMatrix);
    mat4.invert(skyboxViewProjectionInverse, skyboxViewProjectionMatrix);

    app.clear();

    app.disable(PicoGL.DEPTH_TEST);
    app.disable(PicoGL.CULL_FACE);
    skyboxDrawCall.uniform("viewProjectionInverse", skyboxViewProjectionInverse);
    skyboxDrawCall.draw();

    app.enable(PicoGL.DEPTH_TEST);
    //app.enable(PicoGL.CULL_FACE);
    drawCall.uniform("modelViewProjectionMatrix", modelViewProjectionMatrix);
    drawCall.uniform("cameraPosition", cameraPosition);
    drawCall.uniform("modelMatrix", modelMatrix);
    drawCall.uniform("normalMatrix", mat3.normalFromMat4(mat3.create(), modelMatrix));
    drawCall.draw();
}








function drawMirror() {
    mat4.multiply(mirrorModelViewProjectionMatrix, viewProjMatrix, mirrorModelMatrix);
    mirrorDrawCall.uniform("modelViewProjectionMatrix", mirrorModelViewProjectionMatrix);
    mirrorDrawCall.uniform("screenSize", vec2.fromValues(app.width, app.height))
    mirrorDrawCall.draw();
}



let postDrawCall = app.createDrawCall(postProgram, postArray)
    .texture("tex", colorTarget)
    .texture("depthTex", depthTarget)
    .texture("noiseTex", app.createTexture2D(await loadTexture("noise.png")));





function draw(timems) {
    requestAnimationFrame(draw);
    let time = timems / 1000;

    mat4.perspective(projMatrix, Math.PI / 4, app.width / app.height, 0.1, 100.0);
    vec3.rotateY(cameraPosition, vec3.fromValues(0, 2, 4), vec3.fromValues(2, 0, 0), time * 0.05);
    mat4.lookAt(viewMatrix, cameraPosition, vec3.fromValues(0, -0.5, 0), vec3.fromValues(0, 1, 0));

    for (let i = 0; i < numberOfPointLights; i++) {
        vec3.rotateZ(pointLightPositions[i], pointLightInitialPositions[i], vec3.fromValues(0, 0, 0), time);
        positionsBuffer.set(pointLightPositions[i], i * 1);
        colorsBuffer.set(pointLightColors[i], i * 1);
    }

    drawCall.uniform("lightPositions[0]", positionsBuffer);
    drawCall.uniform("lightColors[0]", colorsBuffer);
    app.drawFramebuffer(buffer);

    app.clear();
    drawCall.draw();

    //object itself
    drawCall.uniform(vec4.fromValues(1, 1.0, 1.0, 1.0));
    mat4.fromXRotation(rotateXMatrix, 4.7);
    mat4.multiply(modelViewMatrix, viewMatrix, modelMatrix);
    mat4.multiply(modelMatrix, rotateXMatrix, rotateYMatrix);
    mat4.multiply(modelViewProjectionMatrix, viewProjMatrix, modelMatrix);
    drawCall.draw();

    drawCall.uniform(vec4.fromValues(1, 1.0, 1.0, 1.0));
    mat4.fromXRotation(rotateXMatrix, 4.7);
    mat4.scale(modelMatrix, modelMatrix, vec3.fromValues(1, 0.9, 1));
    mat4.multiply(modelViewMatrix, viewMatrix, modelMatrix);
    mat4.multiply(modelMatrix, rotateXMatrix, rotateYMatrix);
    mat4.multiply(modelViewProjectionMatrix, viewProjMatrix, modelMatrix);
    drawCall.draw();

    mat4.fromXRotation(rotateXMatrix, 0);
    mat4.fromYRotation(rotateYMatrix, 0);
    mat4.mul(mirrorModelMatrix, rotateYMatrix, rotateXMatrix);
    mat4.translate(mirrorModelMatrix, mirrorModelMatrix, vec3.fromValues(0, -1, 0));

    app.clear();
    drawCall.draw();

    app.drawFramebuffer(buffer);
    app.viewport(0, 0, colorTarget.width, colorTarget.height);

    

    app.enable(PicoGL.DEPTH_TEST);
    app.enable(PicoGL.CULL_FACE);
    drawCall.uniform("modelViewProjectionMatrix", modelViewProjectionMatrix);
    renderReflectionTexture();
    drawObjects(cameraPosition, viewMatrix);
    drawMirror();


    postDrawCall.uniform("time", time);
    //postDrawCall.draw();

    requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
