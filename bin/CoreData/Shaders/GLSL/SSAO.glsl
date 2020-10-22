#include "Uniforms.glsl"
#include "Samplers.glsl"
#include "Transform.glsl"
#include "ScreenPos.glsl"
#include "PostProcess.glsl"

varying vec2 vTexCoord;
varying vec4 vScreenPos;
varying vec3 vFarRay;

#ifdef COMPILEPS
    uniform sampler2D sRnd0; // Random noise
    uniform sampler2D sPos1; // Vertex Position (in view space)

    uniform vec2 cSSAOInvSize;
    uniform vec2 cBlurHInvSize;

    uniform float cRadius;
    uniform float cScale;
    uniform float cBias;
    uniform float cStrength;
    uniform vec2 cNoiseScale;

    uniform vec2 cBlurDir;
    uniform float cSharpness;

    float GetAO(const vec3 pos)
    {
        vec4 uv = vec4(pos, 1.0);
        uv = uv * (cViewInvPS * cViewProjPS);
        uv.xyz /= uv.w;
        uv.xyz = uv.xyz * 0.5 + 0.5;
        float d = texture2D(sPos1, uv.xy).z;
        float diff = max(pos.z - d, 0);
        float rangeCheck = 1.0 - smoothstep(cBias, cScale, diff);
        return step(cBias, diff) * rangeCheck;
    }

    vec3 GetNrm(const vec3 p0)
    {
        vec3 p1X = texture2D(sPos1, vTexCoord+vec2(cSSAOInvSize.x, 0.0)).rgb - p0;
        vec3 p2X = p0 - texture2D(sPos1, vTexCoord-vec2(cSSAOInvSize.x, 0.0)).rgb;

        vec3 p1Y = texture2D(sPos1, vTexCoord+vec2(0.0, cSSAOInvSize.y)).rgb - p0;
        vec3 p2Y = p0 - texture2D(sPos1, vTexCoord-vec2(0.0, cSSAOInvSize.y)).rgb;

        vec3 pX = mix(p1X, p2X, abs(min(p1X.z, p2X.z)));
        vec3 pY = mix(p1Y, p2Y, abs(min(p1Y.z, p2Y.z)));
        vec3 n = cross(pX, pY);

        return -normalize(n);
    }

    const float KERNEL_RADIUS = 3;

    float BlurFunction(vec2 uv, float r, float center_c, float center_d, inout float w_total)
    {
        vec2  aoz = texture2D(sDiffMap, uv ).xy;
        float c = aoz.x;
        float d = aoz.y;

        const float BlurSigma = float(KERNEL_RADIUS) * 0.5;
        const float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);

        float ddiff = (d - center_d) * cSharpness;
        float w = exp2(-r*r*BlurFalloff - ddiff*ddiff);
        w_total += w;

        return c*w;
    }

    const int samples = 16;
    vec3 kernel[samples] = vec3[](
        vec3( 0.5381, 0.1856,-0.4319), vec3( 0.1379, 0.2486, 0.4430),
        vec3( 0.3371, 0.5679,-0.0057), vec3(-0.6999,-0.0451,-0.0019),
        vec3( 0.0689,-0.1598,-0.8547), vec3( 0.0560, 0.0069,-0.1843),
        vec3(-0.0146, 0.1402, 0.0762), vec3( 0.0100,-0.1924,-0.0344),
        vec3(-0.3577,-0.5301,-0.4358), vec3(-0.3169, 0.1063, 0.0158),
        vec3( 0.0103,-0.5869, 0.0046), vec3(-0.0897,-0.4940, 0.3287),
        vec3( 0.7119,-0.0154,-0.0918), vec3(-0.0533, 0.0596,-0.5411),
        vec3( 0.0352,-0.0631, 0.5460), vec3(-0.4776, 0.2847,-0.0271)
    );

#endif

void VS()
{
    mat4 modelMatrix = iModelMatrix;
    vec3 worldPos = GetWorldPos(modelMatrix);
    gl_Position = GetClipPos(worldPos);
    vTexCoord = GetQuadTexCoord(gl_Position);
    vScreenPos = GetScreenPos(gl_Position);
    vFarRay = vec3(
        gl_Position.x / gl_Position.w * cFrustumSize.x,
        gl_Position.y / gl_Position.w * cFrustumSize.y,
        cFrustumSize.z);
}

void PS()
{   
    #if defined(GEN_POS)
        gl_FragColor.rgb = texture2D(sDepthBuffer, vTexCoord).r * vFarRay/vFarRay.z;
    #elif defined(GEN_AO)
        // SSAO code partially based on: http://theorangeduck.com/page/pure-depth-ssao
        vec3 pos = texture2D(sPos1, vTexCoord).rgb;
        vec3 nrm = GetNrm(pos);
        vec3 rnd = normalize(texture2D(sRnd0, vTexCoord / (cNoiseScale * cSSAOInvSize)).xyz);

        float rad = cRadius / cFarClipPS;

        float ao = 0.0;
        for (int i = 0; i < samples; ++i)
        {
            vec3 ray = reflect(kernel[i], rnd);
            ray = pos + sign(dot(ray, nrm)) * ray * rad;

            ao += GetAO(ray) * cStrength;
        }
        ao /= float(samples);
        ao = 1.0-clamp(ao, 0, 1);
        gl_FragColor.rg = vec2(ao, pos.z);
    #elif defined(BLUR)
        // Bilateral Blur code from: https://github.com/nvpro-samples/gl_ssao
        vec2 cntr = texture2D(sDiffMap, vTexCoord).rg;
        float ao = cntr.r;
        float w = 1.0;
        for (float r = 1; r <= KERNEL_RADIUS; ++r)
        {
            vec2 uv = vTexCoord + (cGBufferInvSize * cBlurDir) * r;
            ao += BlurFunction(uv, r, cntr.x, cntr.y, w);  
        }

        for (float r = 1; r <= KERNEL_RADIUS; ++r)
        {
            vec2 uv = vTexCoord - (cGBufferInvSize * cBlurDir) * r;
            ao += BlurFunction(uv, r, cntr.x, cntr.y, w);  
        }
        gl_FragColor.rg = vec2(ao/w, cntr.y);
    #elif defined(APPLY)
        float ao = texture2D(sDiffMap, vTexCoord).r;
        gl_FragColor.rgb = vec3(ao);
    #endif
}