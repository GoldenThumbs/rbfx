#include "Uniforms.glsl"
#include "Samplers.glsl"
#include "Transform.glsl"
#include "ScreenPos.glsl"
#include "PostProcess.glsl"
#include "Constants.glsl"

varying vec2 vTexCoord;
varying vec3 vProjInfo;

void VS()
{
    mat4 modelMatrix = iModelMatrix;
    vec3 worldPos = GetWorldPos(modelMatrix);
    gl_Position = GetClipPos(worldPos);
    vTexCoord = GetQuadTexCoord(gl_Position);
    mat4 projMat = cViewInv * cViewProj;
    vProjInfo.x = projMat[0][0];
    vProjInfo.y = projMat[1][1];
    vProjInfo.z = cFrustumSize.z;
}

#ifdef COMPILEPS
    uniform sampler2D sRnd0; // Random noise
    uniform sampler2D sDep1; // Depth Buffer
    uniform sampler2D sNrm2; // Normal Buffer

    // VVV---Usless texture sampler---VVV
    uniform sampler2D sDep2; // Full-resolution depth (for depth-aware upsampling/blur)

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
        const vec2 kernel[8] = vec2[](
            vec2( 1, 0), vec2(-1, 0),
            vec2( 0, 1), vec2( 0,-1),
            vec2( .7, .7), vec2(-.7, .7),
            vec2( .7,-.7), vec2(-.7,-.7)
        );

        vec3 GetPos(vec2 uv)
        {
            float eyeZ = GetDepth(sDep1, uv) * vProjInfo.z;
            vec2 ndcUV = uv * 2.0 - 1.0;
            vec3 p;
            p.xy = eyeZ * ndcUV / vProjInfo.xy;
            p.z = eyeZ;
            return p;
        }

        vec3 MinDiff(vec3 P, vec3 Pr, vec3 Pl)
        {
          vec3 V1 = Pr - P;
          vec3 V2 = P - Pl;
          return (dot(V1,V1) < dot(V2,V2)) ? V1 : V2;
        }

        vec3 GenNrm(vec3 p0)
        {

            vec3 p1X = GetPos(vTexCoord+vec2(cGBufferInvSize.x, 0.0));
            vec3 p2X = GetPos(vTexCoord-vec2(cGBufferInvSize.x, 0.0));

            vec3 p1Y = GetPos(vTexCoord+vec2(0.0, cGBufferInvSize.y));
            vec3 p2Y = GetPos(vTexCoord-vec2(0.0, cGBufferInvSize.y));

            return -normalize(cross(MinDiff(p0, p1X, p2X), MinDiff(p0, p1Y, p2Y)));
        }
        vec3 GetViewNrm(vec2 uv)
        {
            vec3 wNrm = texelFetch(sNrm2, ivec2(gl_FragCoord.xy * 2.0), 0).xyz * 2.0 - 1.0;
            vec4 vNrm = vec4(wNrm, 0.0) * cViewPS;
            return vNrm.xyz;
        }

        float GetAO(vec2 offset, vec3 p, vec3 n)
        {
            vec3 diff = GetPos(vTexCoord + offset) - p;

            float distSqr = dot(diff, diff);
            float invLength = inversesqrt(distSqr);
            float falloff = 1.0/(1.0 + distSqr);
            float angle = dot(n, diff) * invLength;

            return angle*falloff;
        }
    #elif defined(GEN_BLUR)
        const float KERNEL_RADIUS = 3;
        uniform vec2 cBlurDir;

        float Blur(vec2 offset, float baseD, float r, inout float w_total)
        {
            float ao = texture2D(sDiffMap, vTexCoord + offset).r;
            float d = GetDepth(sDep1, vTexCoord + offset);

            const float BlurSigma = float(KERNEL_RADIUS) * 0.5;
            const float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);

            float ddiff = (baseD-d) * cFarClipPS;
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
        gl_FragColor = d;
    #elif defined(GEN_AO)
        vec3 pos = GetPos(vTexCoord);
        #ifdef GEN_NRM
            vec3 nrm = GenNrm(pos);
        #else
            vec3 nrm = GetViewNrm(vTexCoord);
        #endif
        vec2 rnd = texture2D(sRnd0, vTexCoord / (cNoiseScale * cGBufferInvSize)).xy * 2.0 - 1.0;

        float rad = cRadius/pos.z;
        float ao = 0.0;
        for (lowp int i=0; i<samples; i++)
        {
            vec2 coord0 = reflect(kernel[i], rnd)*rad;
            vec2 coord1 = vec2(coord0.x*0.707 - coord0.y*0.707, coord0.x*0.707 + coord0.y*0.707);

            ao += GetAO(coord0*0.25, pos, nrm)*0.25;
            ao += GetAO(coord1*0.50, pos, nrm)*0.25;
            ao += GetAO(coord0*0.75, pos, nrm)*0.25;
            ao += GetAO(coord1, pos, nrm)*0.25;
        }

        ao /= samples;
        ao = 1.0 - cStrength * ao;
        ao = clamp(ao, 0, 1);

        gl_FragColor = vec4(ao);
    #elif defined(GEN_BLUR)
        float ao = texture2D(sDiffMap, vTexCoord).r;
        float depth = GetDepth(sDep2, vTexCoord);

        float w_total = 1.0;

        for (lowp float r = 1; r <= KERNEL_RADIUS; ++r)
        {
            vec2 offset = cBlurDir * cGBufferInvSize * r;
            ao += Blur( offset, depth, r, w_total);
            ao += Blur(-offset, depth, r, w_total); 
        }

        gl_FragColor = vec4(ao/w_total);
    #endif
}