#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "../mixin.h"
#include "CubicSplineHelpers.h"

/* todo:
bounds / centroid
*/



namespace bez
{
    //using StaticClonable = ed::StaticClonable;

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

    

    class CubicBezierSpline : public ed::StaticClonable<CubicBezierSpline>
    {
    public:
        using thisT = CubicBezierSpline;
        using T = CubicBezierSpline;
        /*using uniquePtrT = std::unique_ptr<thisT>;
        using sharedPtrT = std::shared_ptr<thisT>;*/
        //using thisT::thisT;
        DECLARE_DEFINE_CLONABLE_METHODS(thisT)


        //CubicBezierSpline(const WorldSpace* control_points);
        thisT(const WorldSpace* control_points);
        CubicBezierSpline(
            const Eigen::Vector3f& a,
            const Eigen::Vector3f& b,
            const Eigen::Vector3f& c,
            const Eigen::Vector3f& d);
        CubicBezierSpline(
            Eigen::Vector3f& a,
            Eigen::Vector3f& b,
            Eigen::Vector3f& c,
            Eigen::Vector3f& d);
        float ClosestPointToSpline(const WorldSpace& position, const QuinticSolver* solver, WorldSpace& closest, float& u) const;
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

        // weights for matrix representation
        static const Eigen::Matrix4f matrixWeights;

        Eigen::ArrayXf uToLengthMap(int nSamples);

        void copyOther(const thisT& other) {
            control_points_ = other.control_points_;
            polynomial_form_ = other.polynomial_form_;
            derivative_ = other.derivative_;
            precomputed_coefficients_ = other.precomputed_coefficients_;
            inv_leading_coefficient_ = other.inv_leading_coefficient_;
        }

        Eigen::Matrix<float, 4, 3> pointsAsMatrix() {
            /* return a copy, probably best*/
            Eigen::Matrix<float, 4, 3> result;
            
            result.row(0) = toEig(control_points_.at(0));
            result.row(1) = toEig(control_points_.at(1));
            result.row(2) = toEig(control_points_.at(2));
            result.row(3) = toEig(control_points_.at(3));
            return result;
        }

        void transform(Eigen::Affine3f& mat) {
            /* transform this spline in place*/
            control_points_[0] = mat * toEig(control_points_[0]);
            control_points_[1] = mat * toEig(control_points_[1]);
            control_points_[2] = mat * toEig(control_points_[2]);
            control_points_[3] = mat * toEig(control_points_[3]);
            Initialize();
        }
        CubicBezierSpline transformed(Eigen::Affine3f& mat) {
            /* return a transformed copy of this spline*/
            CubicBezierSpline result = cloneOnStack();
            result.transform(mat);
            return result;
        }
    };


    // A Bezier path is a set of Bezier splines which are connected.
    // (The last control point of a spline, is the first control pont of the next spline.)
    struct CubicBezierPath : public ed::StaticClonable<CubicBezierPath>
    {
    public:
        using thisT = CubicBezierPath;
        using T = CubicBezierPath;

        //thisT& operator=(thisT&& other) {
        //    copyOther(other); return *this;
        //} thisT& operator=(const thisT& other) {
        //    copyOther(other); return *this;
        //}

        std::vector<std::unique_ptr<CubicBezierSpline> > splines_;
        std::unique_ptr<Eigen::ArrayXf> uToLengthMap_ = nullptr;
        std::unique_ptr<ClosestPointSolver> solver_ = nullptr;

        //using thisT::thisT;
        DECLARE_DEFINE_CLONABLE_METHODS(thisT)

        //CubicBezierPath(const WorldSpace* control_points, const int num_points) {
        thisT(const WorldSpace* control_points, const int num_points) {
            int num_splines = num_points / 3;
            for (int i = 0; i < num_splines; ++i)
            {
                splines_.emplace_back(new CubicBezierSpline(&control_points[i * 3]));
            }
        }
        CubicBezierPath(const Eigen::MatrixX3f& control_points) {
            int num_points = static_cast<int>(control_points.rows());
            int num_splines = num_points / 3;
            for (int i = 0; i < num_splines; ++i)
            {
                splines_.emplace_back(new CubicBezierSpline(
                    control_points.row(i * 3).matrix(),
                    control_points.row(i * 3 + 1).matrix(),
                    control_points.row(i * 3 + 2).matrix(),
                    control_points.row(i * 3 + 3).matrix()
                    )
                );
            }
        }
        CubicBezierPath(std::vector < std::unique_ptr<CubicBezierSpline>> splines);
        CubicBezierPath(std::vector < CubicBezierSpline> splines);
        CubicBezierPath(std::vector < CubicBezierPath>& splines);
        WorldSpace ClosestPointToPath(const WorldSpace& position, const ClosestPointSolver* solver, float& u) const;
        Eigen::Vector3f ClosestPointToPath(const WorldSpace& position, const ClosestPointSolver* solver, float& u, Eigen::Vector3f& tan) const;
        WorldSpace ClosestPointToPath(const WorldSpace& position, const ClosestPointSolver* solver) const;
        Eigen::Vector3f ClosestPointToPath(const Eigen::Vector3f& position, const ClosestPointSolver* solver) const;
        //float ClosestU(const Eigen::Vector3f& position, const ClosestPointSolver* solver) const;

        ~CubicBezierPath();

        

        WorldSpace CubicBezierPath::tangentAt(float t) const;
        Eigen::Vector3f CubicBezierPath::tangentAt(float t, Eigen::Vector3f& basePos) const;


        std::pair<int, float> global_to_local_param(float t) const {
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

        Eigen::Vector3f eval(float t) const {
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

        int N_SAMPLES = 50;

        Eigen::ArrayXf& getUToLengthMap(int nSamples) {
            if (uToLengthMap_.get() == nullptr) {
                _buildULengthMap(nSamples);
            }
            return *(uToLengthMap_.get());
        }

        float length() {
            return getUToLengthMap(N_SAMPLES)[getUToLengthMap(N_SAMPLES).size()];
        }

        inline ClosestPointSolver* getSolver() {
            if (solver_ == nullptr) {
                solver_ = std::make_unique<ClosestPointSolver>();
            }
            return solver_.get();
        }

        void transform(Eigen::Affine3f& mat) {
            for (auto& i : splines_) {
                i.get()->transform(mat);
            }
        }

    };
}