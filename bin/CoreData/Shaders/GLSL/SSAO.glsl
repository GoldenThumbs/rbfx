#include "Uniforms.glsl"
#include "Samplers.glsl"
#include "Transform.glsl"
#include "ScreenPos.glsl"
#include "PostProcess.glsl"

varying vec2 vTexCoord;
<<<<<<< HEAD
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

    float GetAO(const vec2 ray, const vec3 pos, const vec3 nrm)
    {
        // AO calculation taken from: https://github.com/nvpro-samples/gl_ssao/blob/master/hbao.frag.glsl#L182
        vec3 diff = texture2D(sPos1, vTexCoord + ray).rgb - pos;
        vec3 v = normalize(diff);
        float VdV = dot(v, v);
        float NdV = dot(nrm, v) * 1.0/sqrt(VdV);

        float rangeCheck = 1.0 - smoothstep(cBias, cScale, length(diff));
        return clamp(NdV - cBias,0,1) * rangeCheck * cStrength;
    }

    vec3 GetNrm(const vec3 p0, out vec4 edgesLRTB)
    {
        // Normal generation based on: https://github.com/GameTechDev/ASSAO/blob/master/Projects/ASSAO/ASSAO/ASSAO.hlsl#L279
        vec3 p1X = texture2D(sPos1, vTexCoord+vec2(cSSAOInvSize.x, 0.0)).rgb - p0;
        vec3 p2X = texture2D(sPos1, vTexCoord-vec2(cSSAOInvSize.x, 0.0)).rgb - p0;

        vec3 p1Y = texture2D(sPos1, vTexCoord+vec2(0.0, cSSAOInvSize.y)).rgb - p0;
        vec3 p2Y = texture2D(sPos1, vTexCoord-vec2(0.0, cSSAOInvSize.y)).rgb - p0;

        vec4 edges = vec4(p1X.z, p2X.z, p1Y.z, p2Y.z);
        vec4 edgesSA = edges + edges.yxwz;
        edges = min( abs( edges ), abs( edgesSA ) );
        edges = clamp( ( 1.3 - edges / (p0.z * 0.040) ) ,0,1);

        vec4 AccNrm = vec4(edges.x*edges.z, edges.z*edges.y, edges.y*edges.w, edges.w*edges.x);

        p1X = normalize(p1X);
        p2X = normalize(p2X);
        p1Y = normalize(p1Y);
        p2Y = normalize(p2Y);

        vec3 normal = vec3(0, 0, -0.0005);
        normal += AccNrm.x * cross(p1X, p1Y);
        normal += AccNrm.y * cross(p1Y, p2X);
        normal += AccNrm.z * cross(p2X, p2Y);
        normal += AccNrm.w * cross(p2Y, p1X);

        edgesLRTB = edges;
        return -normalize(normal);
    }

    const float KERNEL_RADIUS = 4;

    float BlurFunction(vec2 uv, float r, float center_c, float center_d, vec2 center_n, inout float w_total)
    {
        vec4 aozn = texture2D(sDiffMap, uv);
        float c = aozn.x;
        float d = aozn.y;
        vec2 n = aozn.zw * 2.0 - 1.0;

        const float BlurSigma = float(KERNEL_RADIUS) * 0.5;
        const float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);

        float ddiff = (center_d - d) * cSharpness;
        float ndiff = distance(center_n, n) * cSharpness;
        float close = abs(ndiff - ddiff);

        float w = exp2(-r*r*BlurFalloff - close*close);
        w_total += w;

        return c*w;
    }

    const int samples = 4;
    vec2 kernel[samples] = vec2[](
        vec2( 1, 0), vec2(0, 1),
        vec2(-1, 0), vec2(0,-1)
    );
=======
varying vec2 vScreenPos;
varying vec3 vFarRay;

#ifdef COMPILEPS
    #ifdef GEN_POS
        uniform sampler2D sDepth0;
        float GetDepth(const vec2 uv)
        {
            #ifdef HWDEPTH
                return ReconstructDepth(texture2D(sDepth0, uv).r);
            #else
                return DecodeDepth(texture2D(sDepth0, uv).rgb);
            #endif
        }
    #else
        uniform sampler2D sPosition0;
        vec3 GetPos(const vec2 uv)
        {
            return texture2D(sPosition0, uv).xyz;
        }
    #endif

    #ifdef GEN_NRM
        vec3 GenNrm(const vec2 uv)
        {
            vec3 p0 = GetPos(uv);

            vec3 p1X = GetPos(uv+vec2(0.001, 0.0)) - p0;
            vec3 p2X = p0 - GetPos(uv-vec2(0.001, 0.0));

            vec3 p1Y = GetPos(uv+vec2(0.0, 0.001)) - p0;
            vec3 p2Y = p0 - GetPos(uv-vec2(0.0, 0.001));

            vec3 pX = mix(p1X, p2X, abs(min(p1X.z, p2X.z)));
            vec3 pY = mix(p1Y, p2Y, abs(min(p1Y.z, p2Y.z)));
            vec3 n = cross(pX, pY);

            return -normalize(n);
        }
    #elif defined(GEN_AO)
        uniform float cRadius;
        uniform float cScale;
        uniform float cBias;
        uniform float cStrength;

        uniform sampler2D sNormal1;
        uniform sampler2D sRandomVec2;

        float GetAO(const vec2 uv, const vec3 pos, const vec3 nrm)
        {
            //SSAO implementation based on https://www.gamedev.net/articles/programming/graphics/a-simple-and-practical-approach-to-ssao-r2753/
            vec3 diff = GetPos(uv) - pos;
            vec3 v = normalize(diff);
            float d = length(diff) * cScale;
            return max(0.0, dot(nrm, v) - cBias) * (1.0/(1.0+d));
        }

        vec3 GetNrm(const vec2 uv)
        {
            return texture2D(sNormal1, uv).xyz * 2.0 - 1.0;
        }

        const int samples = 8;
        const vec2 kernel[samples] = vec2[samples]
        (
            vec2(1, 0), vec2(-1, 0),
            vec2(0, 1), vec2( 0,-1),
            vec2(0.5, 0.5), vec2(-0.5,-0.5),
            vec2(0.5,-0.5), vec2(-0.5, 0.5)
        );
    #else
        uniform sampler2D sMainCol1;
        uniform sampler2D sAO2;
        #ifdef DEBUG
            uniform sampler2D sNormal3;
        #endif
    #endif
>>>>>>> 70dd607eb... Add SSAO sample

#endif

void VS()
{
    mat4 modelMatrix = iModelMatrix;
    vec3 worldPos = GetWorldPos(modelMatrix);
    gl_Position = GetClipPos(worldPos);
    vTexCoord = GetQuadTexCoord(gl_Position);
<<<<<<< HEAD
    vScreenPos = GetScreenPos(gl_Position);
=======
    vScreenPos = GetScreenPosPreDiv(gl_Position);
>>>>>>> 70dd607eb... Add SSAO sample
    vFarRay = vec3(
        gl_Position.x / gl_Position.w * cFrustumSize.x,
        gl_Position.y / gl_Position.w * cFrustumSize.y,
        cFrustumSize.z);
}

void PS()
<<<<<<< HEAD
{   
    #if defined(GEN_POS)
        gl_FragColor.rgb = texelFetch(sDepthBuffer, ivec2(gl_FragCoord.xy * 2.0), 0).r * vFarRay/vFarRay.z;
    #elif defined(GEN_AO)
        // SSAO code partially based on: https://www.gamedev.net/articles/programming/graphics/a-simple-and-practical-approach-to-ssao-r2753/
        vec3 pos = texture2D(sPos1, vTexCoord).rgb;
        vec4 edgesLRTB;
        vec3 nrm = GetNrm(pos, edgesLRTB);
        vec3 rnd = texture2D(sRnd0, vTexCoord / (cNoiseScale * cSSAOInvSize)).xyz * 2.0 - 1.0;

        float rad = cRadius / (pos.z * vFarRay.z);

        float ao = 0.0;
        for (int i = 0; i < samples; ++i)
        {
            vec2 coord0 = reflect(kernel[i], rnd.xy)*rad;
            vec2 coord1 = vec2(coord0.x*0.707 - coord0.y*0.707, coord0.x*0.707 + coord0.y*0.707);

            ao += GetAO(coord0*0.25, pos, nrm); 
            ao += GetAO(coord1*0.50, pos, nrm); 
            ao += GetAO(coord0*0.75, pos, nrm); 
            ao += GetAO(coord1, pos, nrm); 
        }
        ao /= float(samples) * 4.0;
        ao = 1.0-clamp(ao, 0, 1);
        gl_FragColor = vec4(ao, pos.z, nrm.xy*0.5+0.5);
    #elif defined(BLUR)
        // Bilateral Blur code from: https://github.com/nvpro-samples/gl_ssao/blob/master/hbao_blur.frag.glsl#L48
        vec4 cntr = texture2D(sDiffMap, vTexCoord);
        vec2 nrm = cntr.zw * 2.0 - 1.0;
        float ao = cntr.r;
        float w = 1.0;
        for (float r = 1; r <= KERNEL_RADIUS/2; ++r)
        {
            vec2 offset = (cBlurHInvSize * cBlurDir) * r;
            ao += BlurFunction(vTexCoord + offset, r, cntr.x, cntr.y, nrm, w);
            ao += BlurFunction(vTexCoord - offset, r, cntr.x, cntr.y, nrm, w);  
        }
        gl_FragColor = vec4(ao/w, cntr.yzw);
    #elif defined(APPLY)
        float ao = texture2D(sDiffMap, vTexCoord).r;
        gl_FragColor.rgb = vec3(ao);
=======
{  
    #ifdef GEN_POS
        gl_FragColor.rgb = vFarRay * GetDepth(vTexCoord);
    #elif defined(GEN_NRM)
        gl_FragColor.rgb = GenNrm(vTexCoord) * 0.5 + 0.5;
    #elif defined(GEN_AO)
        vec3 pos = GetPos(vTexCoord);
        vec3 nrm = GetNrm(vTexCoord);
        vec2 rnd = normalize(texture2D(sRandomVec2, (1.0 / cGBufferInvSize) * vTexCoord / 4.0).rg) * 2.0 - 1.0;

        float ao = 0.0;
        float rad = cRadius / pos.z;
        for(int i=0; i < samples; i++)
        {
            vec2 ray = reflect(kernel[i], rnd) * rad;
            ao += GetAO(vTexCoord + ray, pos, nrm);
            ao *= cStrength;
        }
        ao = 1.0 - ao * (1.0 / float(samples));
        ao = clamp(ao, 0, 1);
        gl_FragColor.r = ao;
    #else
        #ifndef DEBUG
            vec3 view = texture2D(sMainCol1, vTexCoord).rgb;
            float ao = texture2D(sAO2, vTexCoord).r;
            gl_FragColor.rgb = view * ao;
        #else
            #ifdef P
                gl_FragColor.rgb = texture2D(sPosition0, vTexCoord).rgb;
            #elif defined(N)
                gl_FragColor.rgb = texture2D(sNormal3, vTexCoord).rgb;
            #else
                gl_FragColor.rgb = texture2D(sAO2, vTexCoord).rgb;
            #endif
        #endif
>>>>>>> 70dd607eb... Add SSAO sample
    #endif
}