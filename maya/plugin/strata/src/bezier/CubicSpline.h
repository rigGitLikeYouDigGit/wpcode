#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "../mixin.h"
#include "CubicSplineHelpers.h"

#include "../AABB.h"

/* todo:
bounds / centroid

REFIT EVERYTHING WITH EIGEN
WorldSpace vector type is doin ma head in
*/



namespace bez
{
    //using StaticClonable = strata::StaticClonable;

    class CubicBezierSpline;
    struct Polynomial5;

    constexpr int ArithmeticSum(int n) { return n * (1 + n) / 2; }


    class QuinticSolver
    {
    public:
        QuinticSolver(float tolerance);
        int Solve(
            const Polynomial5& polynomial,
            std::array<float, 5>& out_roots,
            const float interval_min,
            const float interval_max) const;
    private:
        struct SturmInterval
        {
            SturmInterval()
            {
            }

            SturmInterval(float _min, float _max, int _sign_min, int _sign_max, int _id, int _roots)
                : min(_min), max(_max), sign_min(_sign_min), sign_max(_sign_max), id(_id), expected_roots(_roots)
            {
            }

            float min = 0.0f;
            float max = 1.0f;
            int sign_min = 0; // Sign changes for the minimum bound.
            int sign_max = 1; // Sign changes for the max bound.
            int id = 0; // Id shared between this interval and its sibling.
            int expected_roots = 2; // Total roots expected in this interval and its sibling.
        };

        struct Interval
        {
            float min;
            float max;

            Interval& operator=(const SturmInterval& sturm_interval)
            {
                min = sturm_interval.min;
                max = sturm_interval.max;
                return *this;
            }
        };

        typedef std::array<float, ArithmeticSum(5 + 1)> SturmSequence5;

        float tolerance_;
        uint32 max_divisions_;

        // Memory for the stack of intervals being searched. The goal of this is to avoid any memory allocations
        // during the solver. If the tolerance is low, this could just be put on the stack instead.
        mutable std::vector<SturmInterval> interval_storage_;

        void BuildSturmSequence(const Polynomial5& polynomial, SturmSequence5& out_sturm_polynomials) const;
        int CountSturmSignChanges(const SturmSequence5& sturm_polynomials, const float t) const;
        float SolveBisection(const Polynomial5& polynomial, const float interval_min, const float interval_max) const;
    };

    class ClosestPointSolver
    {
    public:
        ClosestPointSolver();
        ~ClosestPointSolver();

        const QuinticSolver* Get() const;
    private:
        //std::unique_ptr<QuinticSolver> solver_;
        QuinticSolver solver_;
    };

    

    class CubicBezierSpline : public strata::StaticClonable<CubicBezierSpline>
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
        Eigen::Vector3f eval(const float t) const;

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
        ClosestPointEquation precomputed_coefficients_ = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
        float inv_leading_coefficient_ = 1.0f;

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
            /* return a copy, probably best
            TODO: work out how to return ref*/
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

        inline std::pair<Eigen::Vector3f, Eigen::Vector3f> minMaxBounds() {
            return std::make_pair(
                Eigen::Vector3f(
                    std::min({
                        control_points_[0].x,
                        control_points_[1].x,
                        control_points_[2].x,
                        control_points_[3].x
                        }),
                    std::min({
                        control_points_[0].y,
                        control_points_[1].y,
                        control_points_[2].y,
                        control_points_[3].y
                        }),
                    std::min({
                        control_points_[0].z,
                        control_points_[1].z,
                        control_points_[2].z,
                        control_points_[3].z
                        })
                ),
                Eigen::Vector3f(
                    std::max({
                        control_points_[0].x,
                        control_points_[1].x,
                        control_points_[2].x,
                        control_points_[3].x
                        }),
                    std::max({
                        control_points_[0].y,
                        control_points_[1].y,
                        control_points_[2].y,
                        control_points_[3].y
                        }),
                    std::max({
                        control_points_[0].z,
                        control_points_[1].z,
                        control_points_[2].z,
                        control_points_[3].z
                        })
                )
            );
        }
        inline aabb::AABB getAABB() {
            auto p = minMaxBounds();
            return aabb::AABB(p.first, p.second);
        }
    };


    // A Bezier path is a set of Bezier splines which are connected.
    // (The last control point of a spline, is the first control pont of the next spline.)

    /* TODO: remove pointers, fine for path to own all its splines densely
    */
    struct CubicBezierPath //: public strata::StaticClonable<CubicBezierPath>
    {
    public:
        using thisT = CubicBezierPath;
        using T = CubicBezierPath;

        //thisT& operator=(thisT&& other) {
        //    copyOther(other); return *this;
        //} thisT& operator=(const thisT& other) {
        //    copyOther(other); return *this;
        //}

        //std::vector<std::unique_ptr<CubicBezierSpline> > splines_;
        //std::unique_ptr<Eigen::ArrayXf> uToLengthMap_ = nullptr;
        //std::unique_ptr<ClosestPointSolver> solver_ = nullptr;
        std::vector<CubicBezierSpline > splines_;
        Eigen::ArrayXf uToLengthMap_;// = nullptr;
        ClosestPointSolver solver_;// = nullptr;

        //using thisT::thisT;
        //DECLARE_DEFINE_CLONABLE_METHODS(thisT)

        //CubicBezierPath(const WorldSpace* control_points, const int num_points) {
        
        thisT() {}

        thisT(const WorldSpace* control_points, const int num_points) {
            int num_splines = num_points / 3;
            for (int i = 0; i < num_splines; ++i)
            {
                //splines_.emplace_back(new CubicBezierSpline(&control_points[i * 3]));
                splines_.emplace_back(CubicBezierSpline(&(control_points[i * 3])));
            }
        }
        CubicBezierPath(
            const Eigen::MatrixX3f& control_points,
            bool closed = false);
        CubicBezierPath(std::vector < std::unique_ptr<CubicBezierSpline>> splines);
        CubicBezierPath(std::vector < CubicBezierSpline> splines);
        CubicBezierPath(std::vector < CubicBezierPath>& splines);
        WorldSpace ClosestPointToPath(const WorldSpace& position, const ClosestPointSolver* solver, float& u) const;
        Eigen::Vector3f ClosestPointToPath(const WorldSpace& position, const ClosestPointSolver* solver, float& u, Eigen::Vector3f& tan) const;
        WorldSpace ClosestPointToPath(const WorldSpace& position, const ClosestPointSolver* solver) const;
        Eigen::Vector3f ClosestPointToPath(const Eigen::Vector3f& position, const ClosestPointSolver* solver) const;


        Eigen::Vector3f ClosestPointToPath(
            const Eigen::Vector3f& position,
            const ClosestPointSolver* solver,
            float& u) const;

        void Initialize() { // call after directly editing curves
            for (auto& sp : splines_) {
                sp.Initialize();
            }
        }

        Eigen::Vector3f CubicBezierPath::tangentAt(float t) const;
        Eigen::Vector3f CubicBezierPath::tangentAt(float t, Eigen::Vector3f& basePos) const;


        std::pair<int, float> globalToLocalParam(float t) const {
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
            auto idT = globalToLocalParam(t);
            //return toEig(splines_[idT.first].get()->EvaluateAt(idT.second));
            return toEig(splines_[idT.first].EvaluateAt(idT.second)); 
        }

        void _buildULengthMap(int nSamples) {
            /* invalidates any previously stored cache, builds 
            length map for all contained splines*/
            //std::unique_ptr<Eigen::ArrayXf> result = std::make_unique<Eigen::ArrayXf>();
            Eigen::ArrayXf result;// = std::make_unique<Eigen::ArrayXf>();
            //Eigen::ArrayXf& resultRef = *result;
            Eigen::ArrayXf& resultRef = result;
            //result->resize(nSamples * splines_.size());
            result.resize(nSamples * splines_.size());

            float prevMax = 0;
            for (int i = 0; i < static_cast<int>(splines_.size()); i++) {
                //auto splineMap = splines_[i].get()->uToLengthMap(nSamples);
                auto splineMap = splines_[i].uToLengthMap(nSamples);
                for (int n = 0; n < nSamples; n++) {
                    resultRef(i * nSamples + n) = splineMap(n) + prevMax;
                }
                prevMax += splineMap[nSamples - 1];
            }
            //uToLengthMap_.reset(std::move(result.get()));
            uToLengthMap_ = result;
        }

        int N_SAMPLES = 50;

        Eigen::ArrayXf& getUToLengthMap(int nSamples) {
            if (uToLengthMap_.rows() == 0) {
            //if (uToLengthMap_.get() == nullptr) {
                _buildULengthMap(nSamples);
            }
            //return *(uToLengthMap_.get());
            return uToLengthMap_;
        }

        float length() {
            return getUToLengthMap(N_SAMPLES)[getUToLengthMap(N_SAMPLES).size()];
        }

        inline ClosestPointSolver* getSolver() {
            return &solver_;
            //if (solver_ == nullptr) {
            //    solver_ = std::make_unique<ClosestPointSolver>();
            //}
            //return solver_.get();
        }

        void transform(Eigen::Affine3f& mat) {
            for (auto& i : splines_) {
                //i.get()->transform(mat);
                i.transform(mat);
            }
        }

        inline aabb::AABB getAABB() {
            aabb::AABB base = splines_[0].getAABB();
            for (int i = 1; i < splines_.size(); i++) {
                base.merge(base, splines_[i].getAABB());
            }
            return base;
        }

    };
}