# Canny-Edge-and-Harris-Corner-Detector

The **Canny edge detector** is an edge detection operator that uses a multi-stage algorithm to detect edges in images. The stages involved are - smoothening, calculating gradients, nonmaximum suppression and thresholding with hysterysis. Canny Edge detector is based on the fact the sudden intensity changes accross the edge but however intensity does not change along the edge.

![Bycycle](/data/bicycle.bmp)
![Bycycle Edges](/output/edge_detection/bicycle_edges.jpg)

**Harris Corner Detector** is a corner detection operator that is commonly used in computer vision algorithms to extract corners and infer interest point of an image. Harris corner detector is based on the fact that around a corner, intensity changes along every direction. This is captured by considereng a window around a point and observing in intensity changes by moving window accross different direction.

![Plane](/data/plane.bmp)
![Plane Corners](/output/corner_detection/plane_corners.jpg)
