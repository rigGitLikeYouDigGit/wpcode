#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "CubicSplineHelpers.h"

/* todo:
bounds / centroid
*/

namespace bez
{
    class CubicBezierSpline;
    class QuinticSolver;

    class ClosestPointSolver
    {
    public:
        ClosestPointSolver();
        ~ClosestPointSolver();

        const QuinticSolver* Get() const;
    private:
        std::unique_ptr<QuinticSolver> solver_;
    };

    

    class CubicBezierSpline
    {
    public:
        CubicBezierSpline(const WorldSpace* control_points);
        float ClosestPointToSpline(const WorldSpace& position, const QuinticSolver* solver, WorldSpace& closest) const;
        WorldSpace EvaluateAt(const float t) const;
        Eigen::Vector3f eval(const float t) const {
            return toEig(EvaluateAt(t));
        }

        WorldSpace tangentAt(float t);

        //private:
        void Initialize();

        typedef std::array<float, 6> ClosestPointEquation;

        std::array<WorldSpace, 4> control_points_;
        std::array<WorldSpace, 4> polynomial_form_; // Coefficents derived from the control points.
        std::array<WorldSpace, 3> derivative_;
        // The closest projected point equation for a given position p, is:
        // Dot(p, derivative_) - Dot(polynomial_form_, derivative) = 0
        // precomputed_coefficients_ stores -Dot(polynomial_form_, derivative) so that only
        // Dot(p, derivative_) needs to be computed for each position.
        ClosestPointEquation precomputed_coefficients_;
        float inv_leading_coefficient_;

        Eigen::VectorXf uToLengthMap(int nSamples);
    };


    // A Bezier path is a set of Bezier splines which are connected.
    // (The last control point of a spline, is the first control pont of the next spline.)
    class CubicBezierPath
    {
    public:
        CubicBezierPath();
        CubicBezierPath(const WorldSpace* control_points, const int num_points);
        CubicBezierPath(std::vector < std::unique_ptr<CubicBezierSpline>> splines);
        WorldSpace ClosestPointToPath(const WorldSpace& position, const ClosestPointSolver* solver) const;

        ~CubicBezierPath();
        std::vector<std::unique_ptr<CubicBezierSpline> > splines_;
        std::unique_ptr<Eigen::ArrayXf> uToLengthMap_ = nullptr;
        

        WorldSpace CubicBezierPath::tangentAt(float t);

        std::pair<int, float> global_to_local_param(float t) {
            int number_of_curves = static_cast<int>(splines_.size());
            int curve_index;
            if (t == 1) {
                curve_index = number_of_curves - 1;
            }
            else {
                curve_index = static_cast<int>(std::floor(number_of_curves * t));
            }
            float curve_fraction = curve_index / (float)number_of_curves;
            return std::make_pair(curve_index, (t - curve_fraction) * number_of_curves);
        }

        Eigen::Vector3f eval(float t) {
            auto idT = global_to_local_param(t);
            return toEig(splines_[idT.first].get()->EvaluateAt(t));
        }

        void _buildULengthMap(int nSamples) {
            /* invalidates any previously stored cache, builds 
            length map for all contained splines*/
            std::unique_ptr<Eigen::ArrayXf> result = std::make_unique<Eigen::ArrayXf>();
            Eigen::ArrayXf& resultRef = *result;
            result->resize(nSamples * splines_.size());

            float prevMax = 0;
            for (int i = 0; i < static_cast<int>(splines_.size()); i++) {
                auto splineMap = splines_[i].get()->uToLengthMap(nSamples);
                for (int n = 0; n < nSamples; n++) {
                    resultRef(i * nSamples + n) = splineMap(n) + prevMax;
                }
                prevMax += splineMap[nSamples - 1];
            }
            uToLengthMap_.reset(std::move(result.get()));
        }

        Eigen::ArrayXf* getUToLengthMap(int nSamples) {
            if (uToLengthMap_.get() == nullptr) {
                _buildULengthMap(nSamples);
            }
            return uToLengthMap_.get();
        }

    };
}