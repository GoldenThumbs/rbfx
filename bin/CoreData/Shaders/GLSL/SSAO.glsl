#include "Uniforms.glsl"
#include "Samplers.glsl"
#include "Transform.glsl"
#include "ScreenPos.glsl"
#include "PostProcess.glsl"
#include "Constants.glsl"

varying vec2 vTexCoord;
varying vec3 vFrustumSize;

void VS()
{
    mat4 modelMatrix = iModelMatrix;
    vec3 worldPos = GetWorldPos(modelMatrix);
    gl_Position = GetClipPos(worldPos);
    vTexCoord = GetQuadTexCoord(gl_Position);
    vFrustumSize = cFrustumSize;
}

#ifdef COMPILEPS
    uniform sampler2D sRnd0; // Random noise
    uniform sampler2D sDep1; // Depth
    uniform sampler2D sNrm2; // Normals

    float GetDepth(vec2 uv)
    {
        #ifdef HWDEPTH
            return ReconstructDepth(texture2D(sDep1, uv).r);
        #else
            return DecodeDepth(texture2D(sDep1, uv).rgb);
        #endif
    }

    float GetFullDepth(vec2 uv)
    {
        #ifdef HWDEPTH
            return ReconstructDepth(texture2D(sDepthBuffer, uv).r);
        #else
            return DecodeDepth(texture2D(sDepthBuffer, uv).rgb);
        #endif
    }

    #if defined(SCALE_DEPTH)
        uniform float cDownscale;
    #elif defined(GEN_AO)
        uniform vec2 cSSAOInvSize;

        uniform float cRadius;
        uniform float cScale;
        uniform float cBias;
        uniform float cStrength;
        uniform vec2 cNoiseScale;

        const int samples = 8;
        const int steps = 4;
        //const vec2 kernel[samples] = vec2[](
        //    vec2( 1, 0), vec2( 0, 1),
        //    vec2(-1, 0), vec2( 0,-1)
        //);

        vec3 GetPos(vec2 uv, float depth)
        {
            vec3 frustum = vFrustumSize;
            vec3 vray = vec3((uv.xy*2.0-1.0)*frustum.xy, frustum.z);
            return depth * vray;
        }

        float GetAO(const vec2 ray, const vec3 pos, const vec3 nrm)
        {
            // AO calculation taken from: https://github.com/nvpro-samples/gl_ssao/blob/master/hbao.frag.glsl#L182
            vec2 uv = vTexCoord+ray;
            vec3 diff = GetPos(uv, GetDepth(uv)) - pos;
            vec3 v = normalize(diff); 

            float VdV = dot(v, v);
            float NdV = dot(nrm, v) * 1.0/sqrt(VdV);

            float d = length(diff) * cScale;
            float rangeCheck = 1.0-smoothstep(0.0, cScale, d);
            return max(0, NdV-cBias) * rangeCheck * cStrength;
        }

        vec3 GetNrmSmpl(const vec3 p0)
        {
            vec3 p1X = GetPos(vTexCoord+vec2(cSSAOInvSize.x, 0.0), GetDepth(vTexCoord+vec2(cSSAOInvSize.x, 0.0))) - p0;
            vec3 p2X = p0 - GetPos(vTexCoord-vec2(cSSAOInvSize.x, 0.0), GetDepth(vTexCoord-vec2(cSSAOInvSize.x, 0.0)));
            vec3 p1Y = GetPos(vTexCoord+vec2(0.0, cSSAOInvSize.y), GetDepth(vTexCoord+vec2(0.0, cSSAOInvSize.y))) - p0;
            vec3 p2Y = p0 - GetPos(vTexCoord-vec2(0.0, cSSAOInvSize.y), GetDepth(vTexCoord-vec2(0.0, cSSAOInvSize.y)));
            return -normalize(cross(min(p1X, p2X), min(p1Y, p2Y)));
        }

        vec3 GetNrmExpn(const vec3 p0)
        {
            // Normal generation based on: https://github.com/GameTechDev/ASSAO/blob/master/Projects/ASSAO/ASSAO/ASSAO.hlsl#L279
            vec3 p1X = GetPos(vTexCoord+vec2(cSSAOInvSize.x, 0.0), GetDepth(vTexCoord+vec2(cSSAOInvSize.x, 0.0))) - p0;
            vec3 p2X = GetPos(vTexCoord-vec2(cSSAOInvSize.x, 0.0), GetDepth(vTexCoord-vec2(cSSAOInvSize.x, 0.0))) - p0;

            vec3 p1Y = GetPos(vTexCoord+vec2(0.0, cSSAOInvSize.y), GetDepth(vTexCoord+vec2(0.0, cSSAOInvSize.y))) - p0;
            vec3 p2Y = GetPos(vTexCoord-vec2(0.0, cSSAOInvSize.y), GetDepth(vTexCoord-vec2(0.0, cSSAOInvSize.y))) - p0;

            //vec4 edges = vec4(p1X.a, p2X.a, p1Y.a, p2Y.a);
            //vec4 edgesSA = edges + edges.yxwz;
            //edges = min( abs( edges ), abs( edgesSA ) );
            //edges = clamp( ( 1.3 - edges / (p0.a * 0.04) ) ,0,1);

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

        vec3 GetNrm(const vec3 p0)
        {
            #ifndef EXPENSIVE_NRM
                return GetNrmSmpl(p0);
            #else
                return GetNrmExpn(p0);
            #endif
        }

    #elif defined(BLUR)
        uniform vec2 cBlurHInvSize;

        uniform vec2 cBlurDir;
        uniform float cSharpness;

        const float KERNEL_RADIUS = 4;

        float BlurFunction(vec2 uv, float r, float center_d, inout float w_total)
        {
            float ao = texture2D(sDiffMap, uv).r;
            float d = GetDepth(uv);

            const float BlurSigma = float(KERNEL_RADIUS) * 0.5;
            const float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);

            float ddiff = (center_d - d) * cSharpness * cFarClipPS;

            float w = exp2(-r*r*BlurFalloff - ddiff*ddiff);
            w_total += w;

            return ao*w;
        }
    #endif

#endif

void PS()
{   
    #if defined(SCALE_DEPTH)

        gl_FragColor.r = texelFetch(sDepthBuffer, ivec2(gl_FragCoord.xy * cDownscale), 0).r;

    #elif defined(GEN_AO)

        vec3 pos = GetPos(vTexCoord, GetDepth(vTexCoord));
        #ifdef GEN_NRM
            vec3 nrm = GetNrm(pos);
        #else
            vec4 nTmp = vec4(texture2D(sNrm2, vTexCoord).xyz * 2.0 - 1.0, 0.0) * cViewPS;
            vec3 nrm = nTmp.xyz;
        #endif
        vec3 rnd = texture2D(sRnd0, vTexCoord / (cNoiseScale * cSSAOInvSize)).xyz;

        float rad = cRadius / pos.z;

        float StepSizePixels = rad / (steps + 1);
        const float Alpha = 2.0 * M_PI / samples;
        float ao = 0.0;
        for (float i = 0; i < samples; ++i)
        {
            float Angle = Alpha * i;
            vec2 Dir = reflect(vec2(cos(Angle), sin(Angle)), rnd.xy);
            float RayPixels = (rnd.z * StepSizePixels + 1.0);

            for (float s = 0; s < steps; ++s)
            {
                vec2 coord0 = round(RayPixels * Dir) * cSSAOInvSize;
                ao += GetAO(coord0, pos, nrm);
                RayPixels += StepSizePixels;
            }
        }
        ao /= float(samples * steps);
        ao = 1.0-clamp(ao * 2, 0, 1);
        gl_FragColor.r = ao;

    #elif defined(BLUR)

        // Bilateral Blur code from: https://github.com/nvpro-samples/gl_ssao/blob/master/hbao_blur.frag.glsl#L48
        float ao = texture2D(sDiffMap, vTexCoord).r;
        float d = GetFullDepth(vTexCoord);
        float w = 1.0;
        for (float r = 1; r <= KERNEL_RADIUS; ++r)
        {
            vec2 offset = (cBlurHInvSize * cBlurDir) * r;
            ao += BlurFunction(vTexCoord + offset, r, d, w);
            ao += BlurFunction(vTexCoord - offset, r, d, w);  
        }
        gl_FragColor.r = ao/w;

    #elif defined(APPLY)

        vec4 view = texture2D(sDiffMap, vTexCoord);
        float ao = texture2D(sDep1, vTexCoord).r;
        gl_FragColor = view * ao;

    #endif
}