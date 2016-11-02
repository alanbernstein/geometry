# geometry

## spline.py
- `slope_controlled_bezier_curve` - define multi-segment bezier curve from list of control knots, each with their own position and direction
- `bezier_curve` - define single bezier curve from control points directly
- `bernstein_poly` - core support function for bezier splines

## curves.py
- `frenet_frame` - compute frenet frame (and other related functions) for a space curve
- `add_frenet_offset` - generate a curve C by applying a variable offset vector F to a space curve S, using the frenet basis (T, N, B) as the x-y-z space for F
- `add_frenet_offset_2D` - similar to add_frenet_offset, but accepts 2D vector function F
- `naturalize_parameter` - generate a curve C following the same path as curve S, but equally-spaced with respect to the parameter
- `frenet_frame_difference` - deprecated old version of frenet_frame

## surfaces.py
- `extrude` - generate a surface S by extruding a plane curve P (or singly-parameterized family thereof) with respect to a space curve C, using the frenet basis (T, N) of C as the x-y plane for P (not yet implemented)


