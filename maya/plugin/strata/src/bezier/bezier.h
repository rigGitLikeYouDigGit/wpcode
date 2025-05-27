#ifndef BEZIER_BEZIER_H
#define BEZIER_BEZIER_H

#include <vector>
#include "CubicSpline.h"

#include "bezier_curve.h"
#include "composite_bezier_curve.h"
#include "fit_composite_bezier_curve.h"

namespace bez {

	inline void example() {
		//using namespace std;

		std::vector<WorldSpace> control_points;

		control_points.push_back(WorldSpace(74.f, 100.f, 1.3f)); // Spline 1 control_points.push_back(WorldSpace(62.f, 88.f, 0.f)); control_points.push_back(WorldSpace(136.f, 48.f, 0.f)); control_points.push_back(WorldSpace(139.f, 69.f, 8.f)); // Spline 2

		control_points.push_back(WorldSpace(171.f, 127.f, 0.f)); control_points.push_back(WorldSpace(276.f, 159.f, 5.f)); control_points.push_back(WorldSpace(195.f, 155.f, 9.94f)); // Spline 3

		control_points.push_back(WorldSpace(185.f, 155.f, 23.f)); control_points.push_back(WorldSpace(185.f, 205.f, 0.333f)); control_points.push_back(WorldSpace(233.f, 205.f, 0.f));

		/*
		ScopedPointer bezier_path = new CubicBezierPath(&control_points[0], (int)control_points.size());

		ScopedPointer solver = new ClosestPointSolver();

		for (a set of positions) {
			WorldSpace solution = bezier_path->ClosestPointToPath(position, solver); // Do something. }
		}
		*/
	}
}
#endif //BEZIER_BEZIER_H
