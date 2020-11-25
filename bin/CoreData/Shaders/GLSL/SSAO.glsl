#include "Uniforms.glsl"
#include "Samplers.glsl"
#include "Transform.glsl"
#include "ScreenPos.glsl"
#include "PostProcess.glsl"
#include "Constants.glsl"

varying vec2 vTexCoord;
varying vec3 vViewRay;

void VS()
{
    mat4 modelMatrix = iModelMatrix;
    vec3 worldPos = GetWorldPos(modelMatrix);
    gl_Position = GetClipPos(worldPos);
    vTexCoord = GetQuadTexCoord(gl_Position);
    vViewRay = gl_Position.xyz;
}

#ifdef COMPILEPS
    uniform sampler2D sRnd0; // Random noise
    uniform sampler2D sDep1; // Depth (also sometimes position, fuck you)

    // VVV---Usless texture sampler---VVV
    //uniform sampler2D sDep2; // Full-resolution depth (for depth-aware upsampling/blur)

    float GetDepth(sampler2D depthSampler, vec2 uv)
    {
        #ifdef HWDEPTH
            return ReconstructDepth(texture2D(depthSampler, uv).r);
        #else
            return DecodeDepth(texture2D(depthSampler, uv).rgb);
        #endif
    }

    #if defined(GEN_AO)
        uniform vec2 cSSAOInvSize;

        uniform float cRadius;
        uniform float cBias;
        uniform float cStrength;
        uniform vec2 cNoiseScale;

        const int samples = 8;
        const float steps = 3.0;
        const vec2 kernel[samples] = vec2[](
            vec2( 1, 0), vec2(-1, 0),
            vec2( 0, 1), vec2( 0,-1),
            vec2( .7, .7), vec2(-.7, .7),
            vec2( .7,-.7), vec2(-.7,-.7)
        );

        vec3 GetPos(vec2 uv)
        {
            vec3 p = texture2D(sDep1, uv).xyz;
            return p;
        }

        vec3 MinDiff(vec3 P, vec3 Pr, vec3 Pl)
        {
          vec3 V1 = Pr - P;
          vec3 V2 = P - Pl;
          return (dot(V1,V1) < dot(V2,V2)) ? V1 : V2;
        }

        // requires 32 bit floating point precision on position buffer to look ok
        vec3 GetNrm(vec3 p0)
        {
            // Normal generation based on: https://github.com/GameTechDev/ASSAO/blob/master/Projects/ASSAO/ASSAO/ASSAO.hlsl#L279
            vec3 p1X = GetPos(vTexCoord+vec2(cGBufferInvSize.x, 0.0)) - p0;
            vec3 p2X = GetPos(vTexCoord-vec2(cGBufferInvSize.x, 0.0)) - p0;

            vec3 p1Y = GetPos(vTexCoord+vec2(0.0, cGBufferInvSize.y)) - p0;
            vec3 p2Y = GetPos(vTexCoord-vec2(0.0, cGBufferInvSize.y)) - p0;

            vec4 edges = abs(vec4(p1X.z, p2X.z, p1Y.z, p2Y.z));
            edges = clamp( ( 1.3 - edges / (p0.z * 0.06 + 0.1) ) ,0,1);

            vec4 AccNrm = vec4(edges.x*edges.z, edges.z*edges.y, edges.y*edges.w, edges.w*edges.x);

            p1X.xyz = normalize(p1X.xyz);
            p2X.xyz = normalize(p2X.xyz);
            p1Y.xyz = normalize(p1Y.xyz);
            p2Y.xyz = normalize(p2Y.xyz);

            vec3 normal = vec3(0, 0, -0.0005);
            normal += AccNrm.x * cross(p1X.xyz, p1Y.xyz);
            normal += AccNrm.y * cross(p1Y.xyz, p2X.xyz);
            normal += AccNrm.z * cross(p2X.xyz, p2Y.xyz);
            normal += AccNrm.w * cross(p2Y.xyz, p1X.xyz);

            return -normalize(normal);
        }

        //vec3 GetNrm(vec3 p0)
        //{

        //    vec3 p1X = GetPos(vTexCoord+vec2(cGBufferInvSize.x, 0.0));
        //    vec3 p2X = GetPos(vTexCoord-vec2(cGBufferInvSize.x, 0.0));

        //    vec3 p1Y = GetPos(vTexCoord+vec2(0.0, cGBufferInvSize.y));
        //    vec3 p2Y = GetPos(vTexCoord-vec2(0.0, cGBufferInvSize.y));

        //    return -normalize(cross(MinDiff(p0, p1X, p2X), MinDiff(p0, p1Y, p2Y)));
        //}

        float GetAO(vec2 offset, vec3 p, vec3 n, float r)
        {
            vec2 offset0 = r * offset;
            vec2 offset1 = vec2(offset0.x*0.707 - offset0.y*0.707, offset0.x*0.707 + offset0.y*0.707);

            // ao calculation based on: https://frictionalgames.com/2014-01-tech-feature-ssao-and-temporal-blur/
            // and also: https://www.gamedev.net/articles/programming/graphics/a-simple-and-practical-approach-to-ssao-r2753/

            vec3 diff0 = GetPos(vTexCoord + offset0*0.75) - p;
            vec3 diff1 = GetPos(vTexCoord + offset1*0.50) - p;
            vec3 diff2 = GetPos(vTexCoord + offset0*0.25) - p;
            vec3 diff3 = GetPos(vTexCoord + offset1) - p;

            vec4 distSqr = vec4(dot(diff0, diff0),
                                dot(diff1, diff1),
                                dot(diff2, diff2),
                                dot(diff3, diff3));
            
            vec4 invLength = inversesqrt(distSqr);

            vec4 falloff = clamp(1.0 + distSqr * invLength * -(1.0-r), vec4(0.0), vec4(1.0));

            vec4 angle = vec4(dot(n, diff0),
                              dot(n, diff1),
                              dot(n, diff2),
                              dot(n, diff3)) * invLength;

            return dot(max(angle - cBias, vec4(0.0)), falloff) * 0.25;
        }
    #elif defined(GEN_BLUR)
        const float KERNEL_RADIUS = 3;
        uniform vec2 cBlurDir;

        float BoxBlur(vec2 offset, float baseD, float r, inout float w_total)
        {
            float ao = texture2D(sDiffMap, vTexCoord + offset).r;
            float d = GetDepth(sDep1, vTexCoord + offset);

            const float BlurSigma = float(KERNEL_RADIUS) * 0.5;
            const float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);

            float ddiff = abs(d - baseD) * 50;
            float w = exp2(-r*r*BlurFalloff * ddiff*ddiff);
            w_total += w;

            return ao*w;
        }
    #endif

#endif

void PS()
{   
    #if defined(DOWNRES_DEPTH)
        vec4 d = texelFetch(sDep1, ivec2(gl_FragCoord.xy * 2.0), 0);
        vec4 d0 = texelFetch(sDep1, ivec2(gl_FragCoord.xy * 2.0) + ivec2( 1, 0.0), 0);
        vec4 d1 = texelFetch(sDep1, ivec2(gl_FragCoord.xy * 2.0) + ivec2(-1, 0.0), 0);
        vec4 d2 = texelFetch(sDep1, ivec2(gl_FragCoord.xy * 2.0) + ivec2( 0.0, 1), 0);
        vec4 d3 = texelFetch(sDep1, ivec2(gl_FragCoord.xy * 2.0) + ivec2( 0.0,-1), 0);
        gl_FragColor = max(d, max(max(d0, d1), max(d2, d3)));
    #elif defined(GEN_POS)
        vec3 p = vViewRay * GetDepth(sDep1, vTexCoord);
        gl_FragColor.rgb = p;
    #elif defined(GEN_AO)
        vec3 pos = GetPos(vTexCoord);
        vec3 nrm = GetNrm(pos);
        vec2 rnd = texture2D(sRnd0, vTexCoord / (cNoiseScale * cGBufferInvSize)).xy * 2.0 - 1.0;

        float radStep = (cRadius / pos.z) / float(steps + 1);
        float ao = 0.0;
        for (lowp int i=0; i<samples; i++)
        {
            float traceResult = 0.0;
            vec2 coord = reflect(kernel[i], rnd);

            float rad = 0.0;
            for (lowp float r=0; r<steps; ++r)
            {
                rad += radStep;
                float occ = GetAO(coord, pos, nrm, rad);
                traceResult = mix(occ, 1.0, traceResult);
            }
            ao += traceResult;
        }

        ao /= samples;
        ao = 1.0 - cStrength * ao;
        ao = clamp(ao, 0, 1);

        gl_FragColor.rgb = vec3(ao);
    #elif defined(GEN_BLUR)
        float ao = texture2D(sDiffMap, vTexCoord).r;
        float depth = GetDepth(sDep1, vTexCoord);

        float w_total = 1.0;

        for (float r = 1; r <= KERNEL_RADIUS; ++r)
        {
            vec2 offset = cBlurDir * cGBufferInvSize * r;
            ao += BoxBlur(offset, r, depth, w_total);  
        }

        for (float r = 1; r <= KERNEL_RADIUS; ++r)
        {
            vec2 offset = cBlurDir * -cGBufferInvSize * r;
            ao += BoxBlur(offset, r, depth, w_total);  
        }

        gl_FragColor.rgb = vec3(ao/w_total);
    #endif
}