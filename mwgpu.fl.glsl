varying vec3 f_color;

void main(void) {
//  gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    gl_FragColor = vec4(f_color.x, f_color.z, f_color.y, 1.0);

}
