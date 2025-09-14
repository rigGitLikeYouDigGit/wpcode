#include "CubicSpline.h"
//#include "CubicSplineHelpers.inl"

#include <array>
#include <vector>
//#include <math>

#include <Eigen/Dense>

#include <xmmintrin.h>
#include <smmintrin.h>

#include "../libEigen.h"

using namespace Eigen;
//using namespace arma;
//#define USE_SIMD_OPTIMIZATION

namespace bez
{

    // The Sturm method of finding roots uses subdivision. 
    const float kTolerance = 0.00001f;

    // Helper to solve a polynomial of a given degree.
    float EvaluatePolynomial(
        const float* polynomial,
        const int degree,
        const float t)
    {
        float result = 0.f;
        const float* coefficient = polynomial;
        for (int i = 0; i < degree; ++i, ++coefficient)
        {
            result += *coefficient;
            result *= t;
        }
        result += *coefficient;

        return result;
    }

    struct Polynomial5
    {
        std::array<float, 5 + 1> equation;

        // Input: polynomial with a leading coefficient of 1.0.
        float EvaluateNormedEquation(
            const float t) const;
    };

#ifndef USE_SIMD_OPTIMIZATION
    float Polynomial5::EvaluateNormedEquation(
        const float t) const
    {
        return EvaluatePolynomial(&equation[0], 5, t);
    }
#endif // !USE_SIMD_OPTIMIZATION

    //constexpr int ArithmeticSum(int n) { return n * (1 + n) / 2; }

    //class CubicBezierSpline
    //{
    //public:
    //    CubicBezierSpline(const Eigen::Vector3f* control_points);
    //    float getClosestPoint(const Eigen::Vector3f& position, const QuinticSolver* solver, Eigen::Vector3f& closest) const;
    //    Eigen::Vector3f EvaluateAt(const float t) const;

    //    Eigen::Vector3f tangentAt(float t);

    ////private:
    //    void Initialize();

    //    typedef std::array<float, 6> ClosestPointEquation;

    //    std::array<Eigen::Vector3f, 4> control_points_;
    //    std::array<Eigen::Vector3f, 4> polynomial_form_; // Coefficents derived from the control points.
    //    std::array<Eigen::Vector3f, 3> derivative_;
    //    // The closest projected point equation for a given position p, is:
    //    // Dot(p, derivative_) - Dot(polynomial_form_, derivative) = 0
    //    // precomputed_coefficients_ stores -Dot(polynomial_form_, derivative) so that only
    //    // Dot(p, derivative_) needs to be computed for each position.
    //    ClosestPointEquation precomputed_coefficients_;
    //    float inv_leading_coefficient_;
    //};

    //CubicBezierPath::CubicBezierPath() {
    //    /* leave empty for now */
    //}

    /*CubicBezierPath::CubicBezierPath(
        const Eigen::Vector3f* control_points,
        const int num_points)
    {
        int num_splines = num_points / 3;
        for (int i = 0; i < num_splines; ++i)
        {
            splines_.emplace_back(new CubicBezierSpline(&control_points[i * 3]));
        }
    }*/

    Eigen::Vector3f CubicBezierSpline::evalHull(const float t) const {
        /* return point on line between control points*/
        int a, b;
        float u = strata::getArrayIndicesTForU(4, t, a, b);
        return strata::lerp<Eigen::Vector3f, float>(
            Eigen::Vector3f(control_points_[a]), 
            Eigen::Vector3f(control_points_[b]), 
            u);
    }
    Eigen::Vector3f CubicBezierSpline::tangentAtHull(const float t) const {
        int a, b;
        float u = strata::getArrayIndicesTForU(4, t, a, b);
        return Eigen::Vector3f(control_points_[b] - control_points_[a]);
    }

    //template<typename CurveT>
    void CubicBezierSpline::pointsInOtherCurveSpace(
        CubicBezierPath& otherCurve,
        Eigen::MatrixX3f& otherNormals,
        Eigen::VectorXf& otherNormalUs,
        std::array<float, 2> startEndGlobalU,
        Eigen::Matrix<float, 4, 3>& newControlPoints,
        Eigen::Vector<float, 4>& newUParams,
        int paramMode
    ) {
        Eigen::MatrixX3f interpolatedNormals(4, 3);

        for (int i = 0; i < 4; i++) {
            float otherU;
            Eigen::Vector3f otherPos;
            Eigen::Vector3f otherTan;
            Eigen::Vector3f otherN;
            float thisU = (float(i) / 3.0f);
            switch (paramMode) {
            case(K_U_PARAM): {
                otherU = startEndGlobalU[0] + (
                    startEndGlobalU[1] - startEndGlobalU[0]
                    ) * thisU;
                newUParams[i] = otherU;
                otherPos = otherCurve.eval(otherU);
                otherTan = otherCurve.tangentAt(otherU);
                break;
            }
            case K_CLOSEST_POINT: {
                otherPos = otherCurve.getClosestPoint(
                    control_points_[i],
                    otherCurve.getSolver(),
                    otherU, otherTan
                );
                newUParams[i] = otherU;
                break;
            }
            }

            strata::interp1D(otherNormalUs, otherNormals,
                otherU, otherN);

            Eigen::Affine3f otherMat;
            otherMat.fromPositionOrientationScale(
                otherPos,
                Eigen::Quaternionf::FromTwoVectors(
                    otherTan, otherN
                ),
                Eigen::Vector3f(1.0f, 1.0f, 1.0f)
                //Eigen::UniformScaling<float>(1.0f)
            );
            newControlPoints.row(i) = otherMat.inverse() * Eigen::Vector3f(control_points_[i]);
        }
        
    }

    CubicBezierSpline CubicBezierSpline::fromPointsTangents(
        Eigen::Vector3f& posA,
        Eigen::Vector3f tanA,
        Eigen::Vector3f& posB,
        Eigen::Vector3f tanB
    ) {
        /* handle simple autoscaling tangents to create a new simple curve
        for now just take 0.3 of overall distance between points*/
        tanA = tanA.normalized() * (posA - posB).norm() * 0.3f;
        tanB = tanB.normalized() * (posA - posB).norm() * 0.3f;
        return CubicBezierSpline(posA, posA + tanA, posB + tanB, posB);
    }


    CubicBezierPath::CubicBezierPath(
        //std::vector < std::unique_ptr<CubicBezierSpline>> splines) : 
        std::vector < CubicBezierSpline> splines) : 
            //splines_(std::move(splines)) {}
            splines_(splines) {}

    CubicBezierPath::CubicBezierPath(std::vector < CubicBezierPath>& splines) {
        // merge multiple paths into this one
        for (CubicBezierPath& path : splines) {
            for (CubicBezierSpline& spline : path.splines_) {
                //ptr->cloneUnique();
                //splines_.push_back(ptr->cloneUnique());
                splines_.push_back(spline);
            }
        }
    }

    CubicBezierPath::CubicBezierPath(
        const MatrixX3f& control_points,
        bool closed) {
        //CubicBezierPath(const Eigen::Matrix3Xf& control_points) {
        //CubicBezierPath(const Eigen::Matrix<float, 3, -1>& control_points) {

        int num_points = static_cast<int>(control_points.rows());
        //int num_points = static_cast<int>(control_points.rows());

        ////// TEMP TEMP TEMP ////// FIX THISS /////
        //int num_points = 4;
        /////////////////////
        int num_splines = num_points / 3;
        if (!closed) {
            num_splines = num_splines - 1;
        }

        splines_.resize(num_splines);

        for (int i = 0; i < num_splines; i++)
        {
            //CubicBezierSpline spl(
            //    control_points.row(i * 3).matrix(),
            //    control_points.row((i * 3 + 1) % num_points).matrix(),
            //    control_points.row((i * 3 + 2) % num_points).matrix(),
            //    control_points.row((i * 3 + 3) % num_points).matrix()
            //);
            //splines_.emplace_back(spl);

            //splines_.emplace_back(
            //    control_points.row(i * 3).matrix(),
            //    control_points.row((i * 3 + 1) % num_points).matrix(),
            //    control_points.row((i * 3 + 2) % num_points).matrix(),
            //    control_points.row((i * 3 + 3) % num_points).matrix()
            //);
            /*splines_.emplace_back(
                Eigen::Vector3f(control_points.row(i * 3)),
                Eigen::Vector3f(control_points.row((i * 3 + 1))),
                Eigen::Vector3f(control_points.row((i * 3 + 2))),
                Eigen::Vector3f(control_points.row((i * 3 + 3)))
            );*/

            Vector3f a(control_points.row(i * 3));
            Vector3f b(control_points.row((i * 3 + 1)));
            Vector3f c(control_points.row((i * 3 + 2)));
            Vector3f d(control_points.row((i * 3 + 3)));

            //CubicBezierSpline sp(a, b, c, d);
            CubicBezierSpline& sp = splines_[i];
            sp.control_points_[0] = a;
            sp.control_points_[1] = b;
            sp.control_points_[2] = c;
            sp.control_points_[3] = d;
            sp.Initialize();


            // EMPLACE_BACK was resetting the first spline to all 0s?
            // no idea why

            /*splines_.emplace_back(
                a, b, c, d
            );*/
            //splines_.push_back(sp);
            int N = 2;
        }

    }

    //CubicBezierPath::~CubicBezierPath() {}

    //Eigen::Vector3f CubicBezierPath::getClosestPoint(
    //    const Eigen::Vector3f& position,
    //    const ClosestPointSolver* solver,
    //    float& u) const
    //{
    //    Eigen::Vector3f min_position{ 0.f };
    //    float min_dist_sq = std::numeric_limits<float>::max();

    //    // The closest point on the path, is the closest point from the set of closest points to each spline.
    //    Eigen::Vector3f spline_position{ 0.f };
    //    u = 0.0f;
    //    float splineN = 0.0f;
    //    for (const auto& spline : splines_)
    //    {
    //        //const float dist_sq = spline->getClosestPoint(position, solver->Get(), spline_position, u);
    //        const float dist_sq = spline.getClosestPoint(position, solver->Get(), spline_position, u);
    //        if (dist_sq < min_dist_sq)
    //        {
    //            min_dist_sq = dist_sq;
    //            min_position = spline_position;
    //            u = u + splineN;
    //        }
    //        splineN += 1.0f;
    //    }
    //    u /= float(splines_.size()); // normalise u across whole path
    //    return min_position;
    //}


    Vector3f CubicBezierPath::tangentAt(float t) const {
        auto r = globalToLocalParam(t);
        return (1.0f / 0.0001f) * (
            //splines_[r.first].get()->EvaluateAt(std::max(1.0f, r.second + 0.0001f)) -
            //splines_[r.first].get()->EvaluateAt(std::min(0.0f, r.second - 0.0001f)));
            splines_[r.first].eval(std::max(1.0f, r.second + 0.0001f)) -
            splines_[r.first].eval(std::min(0.0f, r.second - 0.0001f)));
    }

    Eigen::Vector3f CubicBezierPath::tangentAt(float t, Eigen::Vector3f& basePos) const {
        /* specialty treatment for 1.0 or 0.0 u values -
        * basePos should be original sampled position on curve
        * NOT NORMALISED
        */
        auto r = globalToLocalParam(t);
        if (t > 0.999) { // sample backwards, negate result vector
            //Eigen::Vector3f tanPos = toEig(splines_[r.first].get()->EvaluateAt(t - 0.0001f));
            Eigen::Vector3f tanPos = (splines_[r.first].eval(t - 0.0001f));
            return (1.0f / 0.0001f) * (basePos - tanPos);
        }
        //Eigen::Vector3f tanPos = toEig(splines_[r.first].get()->EvaluateAt(t + 0.0001f));
        Eigen::Vector3f tanPos = (splines_[r.first].eval(t + 0.0001f));
        return (1.0f / 0.0001f) * tanPos - basePos;

    }

    Eigen::Vector3f CubicBezierPath::evalHull(const float t) const {
        /* return point on line between control points*/
        auto p = globalToLocalParam(t);
        return splines_[p.first].evalHull(p.second);
    }
    Eigen::Vector3f CubicBezierPath::tangentAtHull(const float t) const {
        auto p = globalToLocalParam(t);
        return splines_[p.first].tangentAtHull(p.second);
    }

    Eigen::Vector3f CubicBezierPath::getClosestPoint(
        const Eigen::Vector3f& position,
        const ClosestPointSolver* solver,
        float& u,
        Eigen::Vector3f& tan
        ) const
    {// what is consistency
        Eigen::Vector3f result = (getClosestPoint(position, solver, u));
        // sample curve once more to get tangent // save one sample by reusing orig result
        
        tan = tangentAt(u, result);
        return result;
    }

    Eigen::Vector3f CubicBezierPath::getClosestPoint(
        const Eigen::Vector3f& position,
        const ClosestPointSolver* solver) const
    {
        float u = 0.0;
        return getClosestPoint(position, solver, u);
    }

    //Eigen::Vector3f CubicBezierPath::getClosestPoint(
    //    const Eigen::Vector3f& position,
    //    const ClosestPointSolver* solver) const
    //{   
    //    return (getClosestPoint(Eigen::Vector3f(position), solver));
    //}

    Eigen::Vector3f CubicBezierPath::getClosestPoint(
        const Eigen::Vector3f& position,
        const ClosestPointSolver* solver,
        float& u) const
    {
        Eigen::Vector3f min_position{ 0.f, 0.f, 0.f };
        float min_dist_sq = std::numeric_limits<float>::max();

        // The closest point on the path, is the closest point from the set of closest points to each spline.
        Eigen::Vector3f spline_position{ 0.f, 0.f, 0.f };
        u = 0.0f;
        float splineN = 0.0f;
        for (const auto& spline : splines_)
        {
            //const float dist_sq = spline->getClosestPoint(position, solver->Get(), spline_position, u);
            const float dist_sq = spline.getClosestPoint(position, solver->Get(), spline_position, u);
            if (dist_sq < min_dist_sq)
            {
                min_dist_sq = dist_sq;
                min_position = spline_position;
                u = u + splineN;
            }
            splineN += 1.0f;
        }
        u /= float(splines_.size()); // normalise u across whole path
        return (min_position);
    }




    ClosestPointSolver::ClosestPointSolver() :
        //solver_(new QuinticSolver(kTolerance))
        solver_(kTolerance)
    {}

    ClosestPointSolver::~ClosestPointSolver() {}

    const QuinticSolver* ClosestPointSolver::Get() const
    {
        //return solver_.get();
        return &solver_;
    }

    QuinticSolver::QuinticSolver(float tolerance)
    {
        tolerance_ = tolerance;
        max_divisions_ = 1 + (uint32)log2(1.f / tolerance_);
        interval_storage_.reserve(max_divisions_ * 2);
    }

#ifndef USE_SIMD_OPTIMIZATION
    int QuinticSolver::CountSturmSignChanges(
        const SturmSequence5& sturm_polynomials,
        const float t) const
    {
        int sign_changes = 0;
        int index = ArithmeticSum(5 + 1) - 1;
        const union { float fl_val; uint32 n_val; } sign = { sturm_polynomials[index] };
        uint32 previous = sign.n_val;
        float solution = 0.f;
        for (int i = 1; i <= 5; ++i)
        {
            index -= i + 1;
            const float* seq = &sturm_polynomials[index];
            solution = EvaluatePolynomial(seq, i, t);

            // Increment sign_changes if the sign bit changed.
            union { float fl_val; uint32 n_val; } sign_min = { solution };
            sign_changes += (previous ^ sign_min.n_val) >> 31;
            previous = sign_min.n_val;
        }

        return sign_changes;
    }
#endif

    float QuinticSolver::SolveBisection(
        const Polynomial5& polynomial,
        const float interval_min,
        const float interval_max) const
    {
        // Only intervals of where p(min) < 0 and p(max) > 0 can contain a solution.
        // This is equivalent to the distance function from the given position to
        // the solution having a negative slope at the min and a positive slope at max.
        if (polynomial.EvaluateNormedEquation(interval_min) > 0)
        {
            return NAN;
        }
        if (polynomial.EvaluateNormedEquation(interval_max) < 0)
        {
            return NAN;
        }

        const int max_iterations = max_divisions_;
        float bisection_min = interval_min;
        float bisection_max = interval_max;
        for (int i = 0; i < max_iterations; ++i)
        {
            const float mid = (bisection_max + bisection_min) / 2.f;

            const float r = polynomial.EvaluateNormedEquation(mid);
            const union { float fl_val; uint32 n_val; } r_sign = { r };
            if (r_sign.n_val & 0x80000000u)
            {
                if (r >= -tolerance_)
                {
                    return mid;
                }
                bisection_min = mid;
            }
            else
            {
                if (r <= tolerance_)
                {
                    return mid;
                }
                bisection_max = mid;
            }
        }

        return bisection_min;
    }

    void QuinticSolver::BuildSturmSequence(
        const Polynomial5& polynomial,
        SturmSequence5& out_sturm_polynomials) const
    {
        const int degree = 5;
        std::array<double, ArithmeticSum(degree + 1)> sturm_polys;

        for (int i = 0; i <= degree; ++i)
        {
            sturm_polys[i] = polynomial.equation[i];
            sturm_polys[degree + 1 + i] = (degree - i) * sturm_polys[i]; // derivative
        }

        int index = 6;
        double* element = &sturm_polys[6 + 5]; // start of third polynomial
        for (int i = 2; i <= degree; ++i)
        {
            const int dest_poly_degree = degree - i;
            const double* p1 = &sturm_polys[index];
            const double* p2 = &sturm_polys[index - (dest_poly_degree + 3)];
            index += dest_poly_degree + 2;

            // long polynomial division
            // For the expected case that the leading coefficient of denominator, p1, is non-zero and the
            // numerator, p2, is one degree higher than p1, compute the quotient Ax + B and solve for the
            // remainder in one pass.
            if (p1[0] != 0.0)
            {
                // The quotient coefficients represent the first degree polynomial from dividing p1 by p2.
                const double quotient_coefficeint1 = p2[0] / p1[0];
                const double quotient_coefficeint2 = (p2[1] - quotient_coefficeint1 * p1[1]) / p1[0];
                for (int c = 0; c <= dest_poly_degree; ++c, ++element)
                {
                    double remainder_coefficient = quotient_coefficeint2 * p1[c + 1] - p2[c + 2];
                    if (c != dest_poly_degree)
                    {
                        remainder_coefficient += quotient_coefficeint1 * p1[c + 2];
                    }

                    *element = remainder_coefficient;
                }
            }
            // Begin Fix #1
            // This code has been added to address an issue with 0s while computing the remainder.
            // This section has not been optimized, profiled, or tested thoroughly.
            // See for more details: https://github.com/nrtaylor/CubicSplineClosestPoint/issues/1.
            // Perform long polynomial division in the same way as if you were using a tabular method
            // by hand. Only the last row of division has to be saved at each step.
            else
            {
                int quotient_degree = 1;
                int denominator_degree = dest_poly_degree + 1;
                const int numerator_degree = dest_poly_degree + 2;
                while (quotient_degree < numerator_degree && p1[0] == 0.0) {
                    *element = 0.0;
                    ++element;
                    ++p1;
                    --denominator_degree;
                    ++quotient_degree;
                }
                if (quotient_degree >= numerator_degree)
                {
                    // Copy p1 into element so Sturm sequence does not count extra
                    // sign change.
                    *(element - 1) = p1[0];
                    continue;
                }
                std::array<double, degree> division_result;
                for (int j = 0; j <= numerator_degree; ++j)
                {
                    division_result[j] = p2[j];
                }
                for (int shift = 0; shift <= quotient_degree; ++shift)
                {
                    const double factor = division_result[shift] / p1[0];
                    for (int k = 0; k <= denominator_degree; ++k)
                    {
                        division_result[k + shift] = division_result[k + shift] - p1[k] * factor;
                    }
                }
                // Copy remainder into Sturm sequence.
                for (int m = quotient_degree + 1; m <= numerator_degree; ++m, ++element)
                {
                    *element = -division_result[m];
                }
            }
            // End Fix #1
        }

        for (int i = 0; i < ArithmeticSum(degree + 1); ++i)
        {
            out_sturm_polynomials[i] = (float)sturm_polys[i];
        }
    }

    int QuinticSolver::Solve(
        const Polynomial5& polynomial,
        std::array<float, 5>& out_roots,
        const float interval_min,
        const float interval_max) const
    {
        SturmSequence5 sturm_polynomials;
        BuildSturmSequence(polynomial, sturm_polynomials);

        // Set up the first interval.
        interval_storage_.clear();
        int sign_min = CountSturmSignChanges(sturm_polynomials, interval_min);
        int sign_max = CountSturmSignChanges(sturm_polynomials, interval_max);
        const int total_roots = sign_min - sign_max;
        int id = 0;
        interval_storage_.emplace_back(interval_min, interval_max, sign_min, sign_max, id++, total_roots);

        std::array<Interval, 5> root_intervals;
        int found_roots = 0;

        // Isolate roots
        while (!interval_storage_.empty() && total_roots != found_roots)
        {
            SturmInterval i = interval_storage_.back();
            interval_storage_.pop_back();

            int num_roots = i.sign_min - i.sign_max;

            if (num_roots <= 0)
            {
                if (!interval_storage_.empty() &&
                    interval_storage_.back().id == i.id)
                {
                    i = interval_storage_.back();
                    interval_storage_.pop_back();
                    num_roots = i.expected_roots;
                }
                else
                {
                    continue;
                }
            }

            // Prune sibling intervals based on the results of the current interval.
            if (num_roots == i.expected_roots &&
                !interval_storage_.empty() &&
                interval_storage_.back().id == i.id)
            {
                interval_storage_.pop_back();
            }
            else if (num_roots == i.expected_roots - 1 &&
                !interval_storage_.empty() &&
                interval_storage_.back().id == i.id) // This case was a biggest perf improvemnt.
            {
                root_intervals[found_roots++] = interval_storage_.back();
                interval_storage_.pop_back();
            }

            if (num_roots == 1)
            {
                root_intervals[found_roots++] = i;
            }
            else
            {
                float mid = (i.min + i.max) / 2.f;
                if (mid - i.min <= tolerance_)
                {
                    root_intervals[found_roots++] = i;
                }
                else
                {
                    // Divide the current interval and search deeper.
                    const int sign_mid = CountSturmSignChanges(sturm_polynomials, mid);
                    interval_storage_.emplace_back(i.min, mid, i.sign_min, sign_mid, id, num_roots);
                    interval_storage_.emplace_back(mid, i.max, sign_mid, i.sign_max, id, num_roots);
                    ++id;
                }
            }
        }

        int num_real_roots = 0;
        for (int i = 0; i < found_roots; ++i)
        {
            const Interval& interval = root_intervals[i];
            float root = SolveBisection(polynomial, interval.min, interval.max);
            if (!isnan(root))
            {
                out_roots[num_real_roots++] = root;
            }
        }

        return num_real_roots;
    }

    const Eigen::Matrix4f CubicBezierSpline::matrixWeights{
        {1.0f,     0.f,    0.f,    0.0f},
        { - 3.0f,      3.0f,  0.0f,  0.0f },
        {3.0f,  -6.0f, 3.0f,  0.0f},
        { -1.0f, 3.0f,  -3.0f, 1.0f}
    };

    CubicBezierSpline::CubicBezierSpline(const Eigen::Vector3f* control_points)
    //CubicBezierSpline::thisT::thisT(const Eigen::Vector3f* control_points)
    {
        std::copy(control_points, control_points + 4, control_points_.begin());

        Initialize();
    }

    CubicBezierSpline::CubicBezierSpline(
        const Eigen::Vector3f& a,
        const Eigen::Vector3f& b,
        const Eigen::Vector3f& c,
        const Eigen::Vector3f& d
        )
    {
        //std::copy(control_points, control_points + 4, control_points_.begin());
        control_points_[0] = (a);
        control_points_[1] = (b);
        control_points_[2] = (c);
        control_points_[3] = (d);

        Initialize();
    }
    CubicBezierSpline::CubicBezierSpline(
        Eigen::Vector3f& a,
        Eigen::Vector3f& b,
        Eigen::Vector3f& c,
        Eigen::Vector3f& d) 
    {
        control_points_[0] = (a);
        control_points_[1] = (b);
        control_points_[2] = (c);
        control_points_[3] = (d);

        Initialize();
    }

    void CubicBezierSpline::Initialize()
    {
        Eigen::Vector3f& p0 = control_points_[0];
        Eigen::Vector3f& p1 = control_points_[1];
        Eigen::Vector3f& p2 = control_points_[2];
        Eigen::Vector3f& p3 = control_points_[3];

        // Expanding out the parametric cubic Bezier curver equation.
        Eigen::Vector3f n = -1.f * p0 + 3.f * p1 + -3.f * p2 + p3;
        Eigen::Vector3f r = 3.f * p0 + -6.f * p1 + 3.f * p2;
        Eigen::Vector3f s = -3.f * p0 + 3.f * p1;
        Eigen::Vector3f& v = p0;

        polynomial_form_[0] = n;
        polynomial_form_[1] = r;
        polynomial_form_[2] = s;
        polynomial_form_[3] = v; // p0

        // The derivative which is a quadratic equation.
        Eigen::Vector3f j = 3.f * n;
        Eigen::Vector3f k = 2.f * r;
        Eigen::Vector3f& m = s;

        derivative_[0] = j;
        derivative_[1] = k;
        derivative_[2] = m;

        // - Dot(polynomial_form_, derivative_) divided by the leading coefficient
        inv_leading_coefficient_ = -1.f / Dot(j, n);
        precomputed_coefficients_[0] = 1.0f;
        precomputed_coefficients_[1] = -(Dot(j, r) + Dot(k, n));
        precomputed_coefficients_[2] = -(Dot(j, s) + Dot(k, r) + Dot(m, n));
        precomputed_coefficients_[3] = -(Dot(j, v) + Dot(k, s) + Dot(m, r));
        precomputed_coefficients_[4] = -(Dot(k, v) + Dot(m, s));
        precomputed_coefficients_[5] = -Dot(m, v);
        for (int i = 1; i < 6; ++i)
        {
            precomputed_coefficients_[i] *= inv_leading_coefficient_;
        }
    }


    float CubicBezierSpline::getClosestPoint(
        const Eigen::Vector3f& position,
        const QuinticSolver* solver,
        Eigen::Vector3f& closest,
        float& u) const
    {
        Polynomial5 quintic;
        std::copy(precomputed_coefficients_.begin(), precomputed_coefficients_.end(), quintic.equation.begin());

        for (int i = 0; i < 3; ++i)
        {
            quintic.equation[3 + i] += Dot(position, derivative_[i]) * inv_leading_coefficient_;
        }

        std::array<float, 5> realRoots;
        const int roots = solver->Solve(quintic, realRoots, kTolerance, 1.f - kTolerance);

        // Test the first control point.
        Eigen::Vector3f min_position = control_points_[0];
        //float min_dist_sq = LengthSquared(position - min_position);
        float min_dist_sq = (position - min_position).squaredNorm();
        u = 0.0;

        // Test the roots.
        for (int i = 0; i < roots; ++i)
        {
            const Eigen::Vector3f root_position = EvaluateAt(realRoots[i]);
            //const float root_dist_sq = LengthSquared(position - root_position);
            const float root_dist_sq = (position - root_position).squaredNorm();
            if (root_dist_sq < min_dist_sq)
            {
                min_dist_sq = root_dist_sq;
                min_position = root_position;
                u = realRoots[i];
            }
        }

        // Test the last control point.
        //const float dist_sq = LengthSquared(position - control_points_[3]);
        const float dist_sq = (position - control_points_[3]).squaredNorm();
        if (dist_sq < min_dist_sq)
        {
            min_dist_sq = dist_sq;
            min_position = control_points_[3];
            u = 1.0;
        }

        closest = min_position;
        return min_dist_sq;
    }


    float CubicBezierSpline::getClosestPoint(
        const Eigen::Vector3f& position,
        const QuinticSolver* solver,
        Eigen::Vector3f& closest) const {
        float u = 0.0f;
        return getClosestPoint(
            position,
            solver,
            closest,
            u
        );
    }


    Eigen::Affine3f CubicBezierPath::frame(float u) {
        auto result = Eigen::Affine3f::Identity();
        Vector3f normal(0, 1, 0);
        if (finalNormals_.rows() > 0) {
            /* normals have been built, interpolate here */
            normal = strata::lerpSampleMatrix<float, 3>(finalNormals_, u);
        }
        result.translate(eval(u)).rotate(Eigen::Quaternionf::FromTwoVectors(
            tangentAt(u), normal
        ));
        return result;
    }

    float CubicBezierPath::lengthBetween(float uA, float uB) {
        return strata::lerpSampleScalarArr(getUToLengthMap(N_SAMPLES), uB) -
            strata::lerpSampleScalarArr(getUToLengthMap(N_SAMPLES), uA);
    }

    Eigen::Vector3f CubicBezierSpline::EvaluateAt(
        const float t) const
    {
        // The polynomial for is faster at evaluating than the parametric.
        return t * (polynomial_form_[2] + t * (polynomial_form_[1] + t * polynomial_form_[0])) + polynomial_form_[3];
    }

    Eigen::Vector3f CubicBezierSpline::eval(
        const float t) const
    {
        // The polynomial form is faster at evaluating than the parametric.
        return ( t * (polynomial_form_[2] + t * (polynomial_form_[1] + t * polynomial_form_[0])) + polynomial_form_[3] );
    }

    Eigen::Vector3f CubicBezierSpline::tangentAt(float t) {
        return (1.0f / 0.0001f) * (EvaluateAt(std::max(1.0f, t + 0.0001f)) - EvaluateAt(std::min(0.0f, t - 0.0001f)));
    }

    Eigen::ArrayXf CubicBezierSpline::uToLengthMap(int nSamples) {
        /* return a float array to interpolate a u value to an arc length - 
        arr[u] = arcLength
        
        reciprocal of gives length to u
        */
        Eigen::ArrayXf result(nSamples);
        result(0) = 0.0;
        for (int i = 1; i < nSamples; i++) {
            float t = (0.999f / float(nSamples - 1) * float(i));
            float prevT = (0.999f / float(nSamples - 1) * float(i-1));
            result(i) = result(i-1) + (eval(t) - eval(prevT)).norm();
        }
        return result;
    }

    template<typename CurveT>
    void CubicBezierPath::pointsInOtherCurveSpace(
        CurveT& otherCurve,
        Eigen::MatrixX3f& otherNormals,
        Eigen::VectorXf& otherNormalUs,
        std::array<float, 2> startEndGlobalU,
        Eigen::MatrixX3f& newControlPoints,
        Eigen::VectorXf& newUParams,
        int paramMode = K_U_PARAM
    ) {
        /* get control points of this path in space of another */
        

    }

    float BezierSubPath::mapTo(float t) {
        if (reverse) {
            t = 1.0f - t;
        }
        return strata::remap(t, 0.0f, 1.0f, uBounds[0], uBounds[1]);
    }
    float BezierSubPath::mapFrom(float t) {
        if (reverse) {
            t = 1.0f - t;
        }
        return strata::remap(t, uBounds[0], uBounds[1], 0.0f, 1.0f);
    }

    Affine3f BezierSubPath::frame(float u) {
        return _path.frame(mapTo(u));
    }

    /* how many whole splines are included in this view*/
    int BezierSubPath::nWholeSplines() {
        auto startPair = _path.globalToLocalParam(uBounds[0]);
        int startIndex = startPair.first;
        if (startPair.second > 0.0001) {
            /* if coord starts above start of spline, don't include it*/
            startIndex += 1;
        }
        auto endPair = _path.globalToLocalParam(uBounds[1]);
        int endIndex = endPair.first;
        if (endPair.second < 0.9999) {
            endIndex -= 1;
        }
        return std::max(endIndex - startIndex, 0);
    }
    /* how many splines are included at all */
    int BezierSubPath::nSplines() {
        auto startPair = _path.globalToLocalParam(uBounds[0]);
        auto endPair = _path.globalToLocalParam(uBounds[1]);
        return std::max(endPair.first - startPair.first, 1);
    }
    int BezierSubPath::startSplineIndex() {
        return _path.globalToLocalParam(uBounds[0]).first;
    }
    int BezierSubPath::endSplineIndex() {
        return _path.globalToLocalParam(uBounds[1]).first;
    }


}

/*

a way from scratchapixel to eval dense points on a bezier curve in sequence
with an iterative "fast forwards differences" method
void evalBezierCurveFFD(const uint32_t &divs, const Vec3f &P0, const Vec3f &P1, const Vec3f &P2, const Vec3f &P3, Vec3f *B)
{
#if 1
    float h = 1.f / divs;
    Vec3f b0 = P0;
    Vec3f fph = 3 * (P1 - P0) * h;
    Vec3f fpphh = (6 * P0 - 12 * P1 + 6 * P2) * h * h;
    Vec3f fppphhh = (-6 * P0 + 18 * P1 - 18 * P2 + 6 * P3) * h * h * h;
    B[0] = b0;
    for (uint32_t i = 1; i <= divs; ++i) {
        B[i] = B[i - 1] + fph + fpphh / 2 + fppphhh / 6;
        // update bd, bdd
        fph = fph + fpphh + fppphhh / 2;
        fpphh = fpphh + fppphhh;
    }
#else
    Vec3f b0 = P0;
    Vec3f bd0 = 3 * (P1 - P0);
    Vec3f bdd0 = (6 * P0 - 12 * P1 + 6 * P2);
    Vec3f bddd0 = (-6 * P0 + 18 * P1 - 18 * P2 + 6 * P3);
    for (uint32_t i = 0; i <= divs; ++i) {
        float x = i / (float)divs;
        B[i] = b0 + bd0 * x + bdd0 * x * x / 2 + bddd0 * x * x * x / 6;
    }
#endif
}

void evalBezierPatchFFD(const uint32_t &divs, const Vec3f *controlPoints, Vec3f *&P)
{
    // generate grid
    Vec3f controlPointsV[4][divs + 1];
    for (uint16_t i = 0; i < 4; ++i) {
        evalBezierCurveFFD(divs, controlPoints[i], controlPoints[4 + i],
            controlPoints[8 + i], controlPoints[12 + i], controlPointsV[i]);
    }
    for (uint16_t i = 0; i <= divs; ++i) {
        evalBezierCurveFFD(divs, controlPointsV[0][i], controlPointsV[1][i],
            controlPointsV[2][i], controlPointsV[3][i], P + i * (divs + 1));
    }
}

*/


#ifdef USE_SIMD_OPTIMIZATION
#include "CubicSplineSimd.inl"
#endif // USE_SIMD_OPTIMIZATION