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
    vViewRay = vec3(
        gl_Position.x / gl_Position.w * cFrustumSize.x,
        gl_Position.y / gl_Position.w * cFrustumSize.y,
        cFrustumSize.z);
}

#ifdef COMPILEPS
    uniform sampler2D sRnd0; // Random noise
    uniform sampler2D sDep1; // Depth

    float GetDepth(vec2 uv)
    {
        #ifdef HWDEPTH
            return ReconstructDepth(texture2D(sDep1, uv).r);
        #else
            return DecodeDepth(texture2D(sDep1, uv).rgb);
        #endif
    }

    #if defined(GEN_AO)
        uniform vec2 cSSAOInvSize;

        uniform float cRadius;
        uniform float cScale;
        uniform float cBias;
        uniform float cStrength;
        uniform vec2 cNoiseScale;

        const int samples = 8;
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

        float GetAO(vec2 offset, vec3 p, vec3 n)
        {
            vec3 diff = GetPos(vTexCoord + offset) - p;
            vec3 v = normalize(diff);
            return max(dot(n, v) - cBias, 0.0) * smoothstep(0.0, 1.0, cRadius / abs(diff.z * cScale));
        }
    #elif defined(GEN_BLUR)
        float BoxBlur(vec2 offset, float baseD)
        {
            float ao = texture2D(sDiffMap, vTexCoord + offset).r;
            float d = GetDepth(vTexCoord + offset);
            float ddiff = (baseD - d) * 50.0;
            return ao * exp2(ddiff * ddiff);
        }
    #endif

#endif

void PS()
{   
    #if defined(GEN_POS)
        vec2 uv = gl_FragCoord.xy * cGBufferInvSize;
        gl_FragColor.rgb = vViewRay * GetDepth(uv);
    #elif defined(GEN_AO)
        vec3 pos = GetPos(vTexCoord);
        vec3 nrm = GetNrm(pos);
        vec2 rnd = texture2D(sRnd0, vTexCoord / (cNoiseScale * cGBufferInvSize)).xy * 2.0 - 1.0;

        float rad = cRadius / pos.z;
        float ao = 0.0;
        for (lowp int i=0; i<samples; i++)
        {
            vec2 coord = rad * reflect(kernel[i], rnd);
            vec2 coord2 = vec2(coord.x*0.707 - coord.y*0.707, coord.x*0.707 + coord.y*0.707);
            ao += GetAO(coord*0.25, pos, nrm) * 0.25;
            ao += GetAO(coord*0.50, pos, nrm) * 0.25;
            ao += GetAO(coord*0.75, pos, nrm) * 0.25;
            ao += GetAO(coord, pos, nrm) * 0.25;
        }

        ao /= samples;
        ao = 1.0 - cStrength * ao;
        ao = clamp(ao, 0, 1);

        gl_FragColor.rgb = vec3(ao);
    #elif defined(GEN_BLUR)
        float ao = texture2D(sDiffMap, vTexCoord).r;
        float depth = GetDepth(vTexCoord);
        ao += BoxBlur( vec2(1, 0) * cGBufferInvSize, depth);
        ao += BoxBlur(-vec2(1, 0) * cGBufferInvSize, depth);
        ao += BoxBlur( vec2(0, 1) * cGBufferInvSize, depth);
        ao += BoxBlur(-vec2(0, 1) * cGBufferInvSize, depth);
        ao += BoxBlur( vec2(1, 1) * cGBufferInvSize, depth);
        ao += BoxBlur(-vec2(1, 1) * cGBufferInvSize, depth);
        ao += BoxBlur(vec2(-1, 1) * cGBufferInvSize, depth);
        ao += BoxBlur(vec2( 1,-1) * cGBufferInvSize, depth);
        gl_FragColor.rgb = vec3(ao/8.0);
    #endif
}