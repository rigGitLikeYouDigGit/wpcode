
#pragma once

//#include <span>

#include <math.h>
#include <assert.h>
#include <exception>
#include <stdexcept>
#include <tuple>

#include <maya/MVector.h>
#include <maya/MPoint.h>

#include "macro.h"
#include "lib.h"

// just seems more convenient to use maya types for core maths



/* basic jank nurbs functions for drawing curves, framing,
least squares etc
same for surfaces

ONCE WHOLE SYSTEM WORKS, go back and try and use a proper library for
this, made by people with degrees who know what they're doing



Mostly copied and adapted from tinynurbs by pradeep-pyro on github: 
https://github.com/pradeep-pyro/tinynurbs

I couldn't work out how to link another project in VS ._.

abundant use of std::vector makes me wilt
*/


namespace ed {


    template <typename T> class array2
    {
    public:
        array2() = default;
        array2(const array2<T>& arr) = default;
        array2& operator=(const array2& arr) = default;
        array2(array2<T>&& arr) = default;
        array2& operator=(array2&& arr) = default;
        array2(size_t rows, size_t cols, T default_value = T()) { resize(rows, cols, default_value); }
        array2(size_t rows, size_t cols, const std::vector<T>& arr)
            : rows_(rows), cols_(cols), data_(arr)
        {
            if (arr.size() != rows * cols)
            {
                throw std::runtime_error("Dimensions do not match with size of vector");
            }
        }
        void resize(size_t rows, size_t cols, T val = T())
        {
            data_.resize(rows * cols, val);
            rows_ = rows;
            cols_ = cols;
        }
        void clear()
        {
            rows_ = cols_ = 0;
            data_.clear();
        }
        T operator()(size_t row, size_t col) const
        {
            assert(row < rows_ && col < cols_);
            return data_[row * cols_ + col];
        }
        T& operator()(size_t row, size_t col)
        {
            assert(row < rows_ && col < cols_);
            return data_[row * cols_ + col];
        }
        T operator[](size_t idx) const
        {
            assert(idx < data_.size());
            return data_[idx];
        }
        T& operator[](size_t idx)
        {
            assert(idx < data_.size());
            return data_[idx];
        }
        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        size_t size() const { return data_.size(); }

        std::vector<T> data_;

    private:
        size_t rows_, cols_;
    };


    /**
 * Convert an nd point in homogenous coordinates to an (n-1)d point in cartesian
 * coordinates by perspective division
 * @param[in] pt Point in homogenous coordinates
 * @return Point in cartesian coordinates
 */

    inline MVector homogenousToCartesian(const MPoint& pt)
    {
        return MVector(pt / pt.w);
    }

    /**
     * Convert a list of nd points in homogenous coordinates to a list of (n-1)d points in cartesian
     * coordinates by perspective division
     * @param[in] ptsws Points in homogenous coordinates
     * @param[out] pts Points in cartesian coordinates
     * @param[out] ws Homogenous weights
     */
    template <int nd, typename T>
    inline void homogenousToCartesian(const std::vector<MPoint>& ptsws,
        std::vector<MVector>& pts, std::vector<T>& ws)
    {
        pts.clear();
        ws.clear();
        pts.reserve(ptsws.size());
        ws.reserve(ptsws.size());
        for (int i = 0; i < ptsws.size(); ++i)
        {
            const MPoint& ptw_i = ptsws[i];
            pts.push_back(MVector(ptw_i / ptw_i.w));
            ws.push_back(ptw_i.w);
        }
    }

    /**
     * Convert a 2D list of nd points in homogenous coordinates to cartesian
     * coordinates by perspective division
     * @param[in] ptsws Points in homogenous coordinates
     * @param[out] pts Points in cartesian coordinates
     * @param[out] ws Homogenous weights
     */
    template <int nd, typename T>
    inline void homogenousToCartesian(const array2<MPoint>& ptsws,
        array2<MVector>& pts, array2<T>& ws)
    {
        pts.resize(ptsws.rows(), ptsws.cols());
        ws.resize(ptsws.rows(), ptsws.cols());
        for (int i = 0; i < ptsws.rows(); ++i)
        {
            for (int j = 0; j < ptsws.cols(); ++j)
            {
                const MPoint& ptw_ij = ptsws(i, j);
                T w_ij = ptw_ij[nd - 1];
                pts(i, j) = MVector(ptw_ij / w_ij);
                ws(i, j) = w_ij;
            }
        }
    }

    /**
     * Convert an nd point in cartesian coordinates to an (n+1)d point in homogenous
     * coordinates
     * @param[in] pt Point in cartesian coordinates
     * @param[in] w Weight
     * @return Input point in homogenous coordinates
     */
    template <int nd, typename T>
    inline MPoint cartesianToHomogenous(const MVector& pt, T w)
    {
        MPoint pt(pt * w );
        pt.w = w;
        return pt;
    }

    /**
     * Convert list of points in cartesian coordinates to homogenous coordinates
     * @param[in] pts Points in cartesian coordinates
     * @param[in] ws Weights
     * @return Points in homogenous coordinates
     */
    template <int nd, typename T>
    inline std::vector<MPoint>
        cartesianToHomogenous(const std::vector<MVector>& pts, const std::vector<T>& ws)
    {
        std::vector<MPoint> Cw;
        Cw.reserve(pts.size());
        for (int i = 0; i < pts.size(); ++i)
        {
            Cw.push_back(cartesianToHomogenous(pts[i], ws[i]));
        }
        return Cw;
    }

    /**
     * Convert 2D list of points in cartesian coordinates to homogenous coordinates
     * @param[in] pts Points in cartesian coordinates
     * @param[in] ws Weights
     * @return Points in homogenous coordinates
     */
    template <int nd, typename T>
    inline array2<MPoint> cartesianToHomogenous(const array2<MVector>& pts,
        const array2<T>& ws)
    {
        array2<MPoint> Cw(pts.rows(), pts.cols());
        for (int i = 0; i < pts.rows(); ++i)
        {
            for (int j = 0; j < pts.cols(); ++j)
            {
                Cw(i, j) = cartesianToHomogenous(pts(i, j), ws(i, j));
            }
        }
        return Cw;
    }

    /**
     * Convert an (n+1)d point to an nd point without perspective division
     * by truncating the last dimension
     * @param[in] pt Point in homogenous coordinates
     * @return Input point in cartesian coordinates
     */
    template <int nd, typename T>
    inline MVector truncateHomogenous(const MPoint& pt)
    {
        return MVector(pt);
    }

    /**
     * Compute the binomial coefficient (nCk) using the formula
     * \product_{i=0}^k (n + 1 - i) / i
     */
    inline unsigned int binomial(unsigned int n, unsigned int k)
    {
        unsigned int result = 1;
        if (k > n)
        {
            return 0;
        }
        for (unsigned int i = 1; i <= k; ++i)
        {
            result *= (n + 1 - i);
            result /= i;
        }
        return result;
    }

    /**
     * Check if two numbers are close enough within eps
     * @param[in] a First number
     * @param[in] b Second number
     * @param[in] eps Tolerance for checking closeness
     * @return Whether the numbers are close w.r.t. the tolerance
     */

    //// REPLACE WITH EQ MACRO
    //template <typename T> inline bool close(T a, T b, double eps = std::numeric_limits<T>::epsilon())
    //{
    //    return (std::abs(a - b) < eps) ? true : false;
    //}

    /**
     * Map numbers from one interval to another
     * @param[in] val Number to map to another range
     * @param[in] old_min Minimum value of original range
     * @param[in] old_max Maximum value of original range
     * @param[in] new_min Minimum value of new range
     * @param[in] new_max Maximum value of new range
     * @return Number mapped to new range
     */
    template <typename T> 
    inline T mapToRange(T val, T old_min, T old_max, T new_min, T new_max)
    {
        T old_range = old_max - old_min;
        T new_range = new_max - new_min;
        return (((val - old_min) * new_range) / old_range) + new_min;
    }

    template <typename T> 
    int findSpan(unsigned int degree, const std::vector<T>& knots, T u)
    {
        // index of last control point
        int n = static_cast<int>(knots.size()) - degree - 2;
        assert(n >= 0);

        // For values of u that lies outside the domain
        if (u >= knots[n + 1])
        {
            return n;
        }
        if (u <= knots[degree])
        {
            return degree;
        }

        // Binary search
        // TODO: Replace this with std::lower_bound
        int low = degree;
        int high = n + 1;
        int mid = (int)std::floor((low + high) / 2.0);
        while (u < knots[mid] || u >= knots[mid + 1])
        {
            if (u < knots[mid])
            {
                high = mid;
            }
            else
            {
                low = mid;
            }
            mid = (int)std::floor((low + high) / 2.0);
        }
        return mid;
    }

//
//    /**
// * Find the span of the given parameter in the knot vector.
// * @param[in] degree Degree of the curve.
// * @param[in] knots Knot vector of the curve.
// * @param[in] u Parameter value.
// * @return Span index into the knot vector such that (span - 1) < u <= span
//*/
//    template <typename T> 
//    int findSpan(unsigned int degree, const std::vector<T>& knots, T u)
//    {
//        // index of last control point
//        int n = static_cast<int>(knots.size()) - degree - 2;
//        assert(n >= 0);
//
//        // For values of u that lies outside the domain
//        if (u >= knots[n + 1])
//        {
//            return n;
//        }
//        if (u <= knots[degree])
//        {
//            return degree;
//        }
//
//        // Binary search
//        // TODO: Replace this with std::lower_bound
//        int low = degree;
//        int high = n + 1;
//        int mid = (int)std::floor((low + high) / 2.0);
//        while (u < knots[mid] || u >= knots[mid + 1])
//        {
//            if (u < knots[mid])
//            {
//                high = mid;
//            }
//            else
//            {
//                low = mid;
//            }
//            mid = (int)std::floor((low + high) / 2.0);
//        }
//        return mid;
//    }

    /**
     * Compute a single B-spline basis function
     * @param[in] i The ith basis function to compute.
     * @param[in] deg Degree of the basis function.
     * @param[in] knots Knot vector corresponding to the basis functions.
     * @param[in] u Parameter to evaluate the basis functions at.
     * @return The value of the ith basis function at u.
     */
    template <typename T> T bsplineOneBasis(int i, unsigned int deg, const std::vector<T>& U, T u)
    {
        int m = static_cast<int>(U.size()) - 1;
        // Special case
        if ((i == 0 && close(u, U[0])) || (i == m - deg - 1 && close(u, U[m])))
        {
            return 1.0;
        }
        // Local property ensures that basis function is zero outside span
        if (u < U[i] || u >= U[i + deg + 1])
        {
            return 0.0;
        }
        // Initialize zeroth-degree functions
        std::vector<double> N;
        N.resize(deg + 1);
        for (int j = 0; j <= static_cast<int>(deg); j++)
        {
            N[j] = (u >= U[i + j] && u < U[i + j + 1]) ? 1.0 : 0.0;
        }
        // Compute triangular table
        for (int k = 1; k <= static_cast<int>(deg); k++)
        {
            T saved = (EQ(N[0], 0.0)) ? 0.0 : ((u - U[i]) * N[0]) / (U[i + k] - U[i]);
            for (int j = 0; j < static_cast<int>(deg) - k + 1; j++)
            {
                T Uleft = U[i + j + 1];
                T Uright = U[i + j + k + 1];
                if (EQ(N[j + 1], 0.0))
                {
                    N[j] = saved;
                    saved = 0.0;
                }
                else
                {
                    T temp = N[j + 1] / (Uright - Uleft);
                    N[j] = saved + (Uright - u) * temp;
                    saved = (u - Uleft) * temp;
                }
            }
        }
        return N[0];
    }

    /**
     * Compute all non-zero B-spline basis functions
     * @param[in] deg Degree of the basis function.
     * @param[in] span Index obtained from findSpan() corresponding the u and knots.
     * @param[in] knots Knot vector corresponding to the basis functions.
     * @param[in] u Parameter to evaluate the basis functions at.
     * @return N Values of (deg+1) non-zero basis functions.
     */
    template <typename T>
    std::vector<T> bsplineBasis(unsigned int deg, int span, const std::vector<T>& knots, T u)
    {
        std::vector<T> N;
        N.resize(deg + 1, T(0));
        std::vector<T> left, right;
        left.resize(deg + 1, static_cast<T>(0.0));
        right.resize(deg + 1, static_cast<T>(0.0));
        T saved = 0.0, temp = 0.0;

        N[0] = 1.0;

        for (int j = 1; j <= static_cast<int>(deg); j++)
        {
            left[j] = (u - knots[span + 1 - j]);
            right[j] = knots[span + j] - u;
            saved = 0.0;
            for (int r = 0; r < j; r++)
            {
                temp = N[r] / (right[r + 1] + left[j - r]);
                N[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            N[j] = saved;
        }
        return N;
    }

    /**
     * Compute all non-zero derivatives of B-spline basis functions
     * @param[in] deg Degree of the basis function.
     * @param[in] span Index obtained from findSpan() corresponding the u and knots.
     * @param[in] knots Knot vector corresponding to the basis functions.
     * @param[in] u Parameter to evaluate the basis functions at.
     * @param[in] num_ders Number of derivatives to compute (num_ders <= deg)
     * @return ders Values of non-zero derivatives of basis functions.
     */
    template <typename T>
    array2<T> bsplineDerBasis(unsigned int deg, int span, const std::vector<T>& knots, T u,
        int num_ders)
    {
        std::vector<T> left, right;
        left.resize(deg + 1, 0.0);
        right.resize(deg + 1, 0.0);
        T saved = 0.0, temp = 0.0;

        array2<T> ndu(deg + 1, deg + 1);
        ndu(0, 0) = 1.0;

        for (int j = 1; j <= static_cast<int>(deg); j++)
        {
            left[j] = u - knots[span + 1 - j];
            right[j] = knots[span + j] - u;
            saved = 0.0;

            for (int r = 0; r < j; r++)
            {
                // Lower triangle
                ndu(j, r) = right[r + 1] + left[j - r];
                temp = ndu(r, j - 1) / ndu(j, r);
                // Upper triangle
                ndu(r, j) = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }

            ndu(j, j) = saved;
        }

        array2<T> ders(num_ders + 1, deg + 1, T(0));

        for (int j = 0; j <= static_cast<int>(deg); j++)
        {
            ders(0, j) = ndu(j, deg);
        }

        array2<T> a(2, deg + 1);

        for (int r = 0; r <= static_cast<int>(deg); r++)
        {
            int s1 = 0;
            int s2 = 1;
            a(0, 0) = 1.0;

            for (int k = 1; k <= num_ders; k++)
            {
                T d = 0.0;
                int rk = r - k;
                int pk = deg - k;
                int j1 = 0;
                int j2 = 0;

                if (r >= k)
                {
                    a(s2, 0) = a(s1, 0) / ndu(pk + 1, rk);
                    d = a(s2, 0) * ndu(rk, pk);
                }

                if (rk >= -1)
                {
                    j1 = 1;
                }
                else
                {
                    j1 = -rk;
                }

                if (r - 1 <= pk)
                {
                    j2 = k - 1;
                }
                else
                {
                    j2 = deg - r;
                }

                for (int j = j1; j <= j2; j++)
                {
                    a(s2, j) = (a(s1, j) - a(s1, j - 1)) / ndu(pk + 1, rk + j);
                    d += a(s2, j) * ndu(rk + j, pk);
                }

                if (r <= pk)
                {
                    a(s2, k) = -a(s1, k - 1) / ndu(pk + 1, r);
                    d += a(s2, k) * ndu(r, pk);
                }

                ders(k, r) = d;

                int temp = s1;
                s1 = s2;
                s2 = temp;
            }
        }

        T fac = static_cast<T>(deg);
        for (int k = 1; k <= num_ders; k++)
        {
            for (int j = 0; j <= static_cast<int>(deg); j++)
            {
                ders(k, j) *= fac;
            }
            fac *= static_cast<T>(deg - k);
        }

        return ders;
    }


    // Forward declaration
    template <typename T> struct RationalCurve;

    /**
    Struct for holding a polynomial B-spline curve
    @tparam T Data type of control points and knots (float or double)
    */
    template <typename T> struct Curve
    {
        unsigned int degree;
        std::vector<T> knots;
        std::vector<MVector> control_points;

        Curve() = default;
        Curve(const RationalCurve<T>& crv) : Curve(crv.degree, crv.knots, crv.control_points) {}
        Curve(unsigned int degree, const std::vector<T>& knots,
            const std::vector<MVector>& control_points)
            : degree(degree), knots(knots), control_points(control_points)
        {
        }
    };

    /**
    Struct for holding a rational B-spline curve
    @tparam T Data type of control points and knots (float or double)
    */
    template <typename T> struct RationalCurve
    {
        unsigned int degree;
        std::vector<T> knots;
        std::vector<MVector> control_points;
        std::vector<T> weights;

        RationalCurve() = default;
        RationalCurve(const Curve<T>& crv)
            : RationalCurve(crv, std::vector<T>(crv.control_points.size(), 1.0))
        {
        }
        RationalCurve(const Curve<T>& crv, const std::vector<T>& weights)
            : RationalCurve(crv.degree, crv.knots, crv.control_points, weights)
        {
        }
        RationalCurve(unsigned int degree, const std::vector<T>& knots,
            const std::vector<MVector>& control_points, const std::vector<T> weights)
            : degree(degree), knots(knots), control_points(control_points), weights(weights)
        {
        }
    };

    // Typedefs for ease of use
    typedef Curve<float> Curve3f;
    typedef Curve<double> Curve3d;
    typedef RationalCurve<float> RationalCurve3f;
    typedef RationalCurve<double> RationalCurve3d;


    /* reminder: rational curve is a special case curve where all knots have a weight of 1.0.
    for non-closed curves, rarely useful - most splines are non-rational
    
    */


    // Forward declaration
    template <typename T> struct RationalSurface;

    /**
    Struct for representing a non-rational NURBS surface
    \tparam T Data type of control points and weights (float or double)
    */
    template <typename T> struct Surface
    {
        unsigned int degree_u, degree_v;
        std::vector<T> knots_u, knots_v;
        array2<MVector> control_points;

        Surface() = default;
        Surface(const RationalSurface<T>& srf)
            : degree_u(srf.degree_u), degree_v(srf.degree_v), knots_u(srf.knots_u),
            knots_v(srf.knots_v), control_points(srf.control_points)
        {
        }
        Surface(unsigned int degree_u, unsigned int degree_v, const std::vector<T>& knots_u,
            const std::vector<T>& knots_v, array2<MVector> control_points)
            : degree_u(degree_u), degree_v(degree_v), knots_u(knots_u), knots_v(knots_v),
            control_points(control_points)
        {
        }
    };

    /**
    Struct for representing a non-rational NURBS surface
    \tparam T Data type of control points and weights (float or double)
    */
    template <typename T> struct RationalSurface
    {
        unsigned int degree_u, degree_v;
        std::vector<T> knots_u, knots_v;
        array2<MVector> control_points;
        array2<T> weights;

        RationalSurface() = default;
        RationalSurface(const Surface<T>& srf, const array2<T>& weights)
            : degree_u(srf.degree_u), degree_v(srf.degree_v), knots_u(srf.knots_u),
            knots_v(srf.knots_v), control_points(srf.control_points), weights(weights)
        {
        }
        RationalSurface(const Surface<T>& srf)
            : RationalSurface(srf, array2<T>(srf.control_points.rows(), srf.control_points.cols(), 1.0))
        {
        }
        RationalSurface(unsigned int degree_u, unsigned int degree_v, const std::vector<T>& knots_u,
            const std::vector<T>& knots_v, const array2<MVector>& control_points,
            const array2<T>& weights)
            : degree_u(degree_u), degree_v(degree_v), knots_u(knots_u), knots_v(knots_v),
            control_points(control_points), weights(weights)
        {
        }
    };

    // Typedefs for ease of use
    typedef Surface<float> Surface3f;
    typedef Surface<double> Surface3d;
    typedef RationalSurface<float> RationalSurface3f;
    typedef RationalSurface<double> RationalSurface3d;



    /**
     * Evaluate point on a nonrational NURBS curve
     * @param[in] degree Degree of the given curve.
     * @param[in] knots Knot vector of the curve.
     * @param[in] control_points Control points of the curve.
     * @param[in] u Parameter to evaluate the curve at.
     * @return point Resulting point on the curve at parameter u.
     */
    template <int dim, typename T>
    MVector curvePoint(unsigned int degree, const std::vector<T>& knots,
        const std::vector<MVector>& control_points, T u)
    {
        // Initialize result to 0s
        //glm::vec<dim, T> point(T(0));
        MVector point;

        // Find span and corresponding non-zero basis functions
        int span = findSpan(degree, knots, u);
        std::vector<T> N = bsplineBasis(degree, span, knots, u);

        // Compute point
        for (unsigned int j = 0; j <= degree; j++)
        {
            point += static_cast<T>(N[j]) * control_points[span - degree + j];
        }
        return point;
    }

    /**
     * Evaluate derivatives of a non-rational NURBS curve
     * @param[in] degree Degree of the curve
     * @param[in] knots Knot vector of the curve.
     * @param[in] control_points Control points of the curve.
     * @param[in] num_ders Number of times to derivate.
     * @param[in] u Parameter to evaluate the derivatives at.
     * @return curve_ders Derivatives of the curve at u.
     * E.g. curve_ders[n] is the nth derivative at u, where 0 <= n <= num_ders.
     */
    template <int dim, typename T>
    std::vector<MVector> curveDerivatives(unsigned int degree, const std::vector<T>& knots,
        const std::vector<MVector>& control_points,
        int num_ders, T u)
    {

        //typedef glm::vec<dim, T> tvecn;
        //typedef glm::vec<dim, T> tvecn;
        using std::vector;

        std::vector<MVector> curve_ders;
        curve_ders.resize(num_ders + 1);

        // Assign higher order derivatives to zero
        for (int k = degree + 1; k <= num_ders; k++)
        {
            curve_ders[k] = tvecn(0.0);
        }

        // Find the span and corresponding non-zero basis functions & derivatives
        int span = findSpan(degree, knots, u);
        array2<T> ders = bsplineDerBasis<T>(degree, span, knots, u, num_ders);

        // Compute first num_ders derivatives
        int du = num_ders < static_cast<int>(degree) ? num_ders : static_cast<int>(degree);
        for (int k = 0; k <= du; k++)
        {
            //curve_ders[k] = tvecn(0.0);
            curve_ders[k] = MVector();
            for (int j = 0; j <= static_cast<int>(degree); j++)
            {
                curve_ders[k] += static_cast<T>(ders(k, j)) * control_points[span - degree + j];
            }
        }
        return curve_ders;
    }

    /**
     * Evaluate point on a nonrational NURBS surface
     * @param[in] degree_u Degree of the given surface in u-direction.
     * @param[in] degree_v Degree of the given surface in v-direction.
     * @param[in] knots_u Knot vector of the surface in u-direction.
     * @param[in] knots_v Knot vector of the surface in v-direction.
     * @param[in] control_points Control points of the surface in a 2d array.
     * @param[in] u Parameter to evaluate the surface at.
     * @param[in] v Parameter to evaluate the surface at.
     * @return point Resulting point on the surface at (u, v).
     */
    template <int dim, typename T>
    MVector surfacePoint(unsigned int degree_u, unsigned int degree_v,
        const std::vector<T>& knots_u, const std::vector<T>& knots_v,
        const array2<MVector>& control_points, T u, T v)
    {

        // Initialize result to 0s
        //glm::vec<dim, T> point(T(0.0));
        MVector point;

        // Find span and non-zero basis functions
        int span_u = findSpan(degree_u, knots_u, u);
        int span_v = findSpan(degree_v, knots_v, v);
        std::vector<T> Nu = bsplineBasis(degree_u, span_u, knots_u, u);
        std::vector<T> Nv = bsplineBasis(degree_v, span_v, knots_v, v);

        for (int l = 0; l <= degree_v; l++)
        {
            glm::vec<dim, T> temp(0.0);
            for (int k = 0; k <= degree_u; k++)
            {
                temp += static_cast<T>(Nu[k]) *
                    control_points(span_u - degree_u + k, span_v - degree_v + l);
            }

            point += static_cast<T>(Nv[l]) * temp;
        }
        return point;
    }

    /**
     * Evaluate derivatives on a non-rational NURBS surface
     * @param[in] degree_u Degree of the given surface in u-direction.
     * @param[in] degree_v Degree of the given surface in v-direction.
     * @param[in] knots_u Knot vector of the surface in u-direction.
     * @param[in] knots_v Knot vector of the surface in v-direction.
     * @param[in] control_points Control points of the surface in a 2D array.
     * @param[in] num_ders Number of times to differentiate
     * @param[in] u Parameter to evaluate the surface at.
     * @param[in] v Parameter to evaluate the surface at.
     * @param[out] surf_ders Derivatives of the surface at (u, v).
     */
    template <int dim, typename T>
    array2<MVector> surfaceDerivatives(unsigned int degree_u, unsigned int degree_v, const std::vector<T>& knots_u,
            const std::vector<T>& knots_v, const array2<MVector>& control_points,
            unsigned int num_ders, T u, T v)
    {

        array2<MVector> surf_ders(num_ders + 1, num_ders + 1, MVector());

        // Set higher order derivatives to 0
        for (int k = degree_u + 1; k <= num_ders; k++)
        {
            for (int l = degree_v + 1; l <= num_ders; l++)
            {
                surf_ders(k, l) = MVector();
            }
        }

        // Find span and basis function derivatives
        int span_u = findSpan(degree_u, knots_u, u);
        int span_v = findSpan(degree_v, knots_v, v);
        array2<T> ders_u = bsplineDerBasis(degree_u, span_u, knots_u, u, num_ders);
        array2<T> ders_v = bsplineDerBasis(degree_v, span_v, knots_v, v, num_ders);

        // Number of non-zero derivatives is <= degree
        unsigned int du = std::min(num_ders, degree_u);
        unsigned int dv = std::min(num_ders, degree_v);

        std::vector<MVector> temp;
        temp.resize(degree_v + 1);
        // Compute derivatives
        for (int k = 0; k <= du; k++)
        {
            for (int s = 0; s <= degree_v; s++)
            {
                temp[s] = MVector();
                for (int r = 0; r <= degree_u; r++)
                {
                    temp[s] += static_cast<T>(ders_u(k, r)) *
                        control_points(span_u - degree_u + r, span_v - degree_v + s);
                }
            }

            int dd = std::min(num_ders - k, dv);

            for (int l = 0; l <= dd; l++)
            {
                for (int s = 0; s <= degree_v; s++)
                {
                    surf_ders(k, l) += ders_v(l, s) * temp[s];
                }
            }
        }
        return surf_ders;
    }


/////////////////////////////////////////////////////////////////////

/**
Evaluate point on a nonrational NURBS curve
@param[in] crv Curve object
@param[in] u Parameter to evaluate the curve at.
@return point Resulting point on the curve at parameter u.
*/
template <typename T> 
MVector curvePoint(const Curve<T>& crv, T u)
{
    return curvePoint(crv.degree, crv.knots, crv.control_points, u);
}

/**
 * Evaluate point on a rational NURBS curve
 * @param[in] crv RationalCurve object
 * @param[in] u Parameter to evaluate the curve at.
 * @return point Resulting point on the curve.
 */
template <typename T> MVector curvePoint(const RationalCurve<T>& crv, T u)
{

    typedef glm::vec<4, T> tvecnp1;

    // Compute homogenous coordinates of control points
    std::vector<tvecnp1> Cw;
    Cw.reserve(crv.control_points.size());
    for (size_t i = 0; i < crv.control_points.size(); i++)
    {
        Cw.push_back(tvecnp1(util::cartesianToHomogenous(crv.control_points[i], crv.weights[i])));
    }

    // Compute point using homogenous coordinates
    tvecnp1 pointw = curvePoint(crv.degree, crv.knots, Cw, u);

    // Convert back to cartesian coordinates
    return util::homogenousToCartesian(pointw);
}

/**
 * Evaluate derivatives of a non-rational NURBS curve
 * @param[in] crv Curve object
 * @param[in] num_ders Number of times to derivate.
 * @param[in] u Parameter to evaluate the derivatives at.
 * @return curve_ders Derivatives of the curve at u.
 * E.g. curve_ders[n] is the nth derivative at u, where 0 <= n <= num_ders.
 */
template <typename T>
std::vector<MVector> curveDerivatives(const Curve<T>& crv, int num_ders, T u)
{
    return curveDerivatives(crv.degree, crv.knots, crv.control_points, num_ders, u);
}

/**
 * Evaluate derivatives of a rational NURBS curve
 * @param[in] u Parameter to evaluate the derivatives at.
 * @param[in] knots Knot vector of the curve.
 * @param[in] control_points Control points of the curve.
 * @param[in] weights Weights corresponding to each control point.
 * @param[in] num_ders Number of times to differentiate.
 * @param[inout] curve_ders Derivatives of the curve at u.
 * E.g. curve_ders[n] is the nth derivative at u, where n is between 0 and
 * num_ders-1.
 */
template <typename T>
std::vector<MVector> curveDerivatives(const RationalCurve<T>& crv, int num_ders, T u)
{

    /*typedef glm::vec<3, T> tvecn;
    typedef glm::vec<4, T> tvecnp1;*/

    std::vector<MVector> curve_ders;
    curve_ders.reserve(num_ders + 1);

    // Compute homogenous coordinates of control points
    std::vector<MPoint> Cw;
    Cw.reserve(crv.control_points.size());
    for (size_t i = 0; i < crv.control_points.size(); i++)
    {
        Cw.push_back(cartesianToHomogenous(crv.control_points[i], crv.weights[i]));
    }

    // Derivatives of Cw
    std::vector<MPoint> Cwders = curveDerivatives(crv.degree, crv.knots, Cw, num_ders, u);

    // Split Cwders into coordinates and weights
    std::vector<MVector> Aders;
    std::vector<T> wders;
    for (const auto& val : Cwders)
    {
        Aders.push_back(truncateHomogenous(val));
        wders.push_back(val.w);
    }

    // Compute rational derivatives
    for (int k = 0; k <= num_ders; k++)
    {
        //tvecn v = Aders[k];
        MVector v = Aders[k];
        for (int i = 1; i <= k; i++)
        {
            v -= static_cast<T>(binomial(k, i)) * wders[i] * curve_ders[k - i];
        }
        curve_ders.push_back(v / wders[0]);
    }
    return curve_ders;
}

/**
 * Evaluate the tangent of a B-spline curve
 * @param[in] crv Curve object
 * @return Unit tangent of the curve at u.
 */
template <typename T> MVector curveTangent(const Curve<T>& crv, T u)
{
    std::vector<MVector> ders = curveDerivatives(crv, 1, u);
    MVector du = ders[1];
    //T du_len = glm::length(du);
    T du_len = T(du.length());
    if (!EQ(du_len, T(0)))
    {
        du /= du_len;
    }
    return du;
}

/**
 * Evaluate the tangent of a rational B-spline curve
 * @param[in] crv RationalCurve object
 * @return Unit tangent of the curve at u.
 */
template <typename T> MVector curveTangent(const RationalCurve<T>& crv, T u)
{
    std::vector<MVector> ders = curveDerivatives(crv, 1, u);
    MVector du = ders[1];
    //T du_len = glm::length(du);
    T du_len = T(du.length());
    if (!EQ(du_len, T(0)))
    {
        du /= du_len;
    }
    return du;
}

/**
 * Evaluate point on a nonrational NURBS surface
 * @param[in] srf Surface object
 * @param[in] u Parameter to evaluate the surface at.
 * @param[in] v Parameter to evaluate the surface at.
 * @return Resulting point on the surface at (u, v).
 */
template <typename T> MVector surfacePoint(const Surface<T>& srf, T u, T v)
{
    return surfacePoint(srf.degree_u, srf.degree_v, srf.knots_u, srf.knots_v,
        srf.control_points, u, v);
}

/**
 * Evaluate point on a non-rational NURBS surface
 * @param[in] srf RationalSurface object
 * @param[in] u Parameter to evaluate the surface at.
 * @param[in] v Parameter to evaluate the surface at.
 * @return Resulting point on the surface at (u, v).
 */
template <typename T> MVector surfacePoint(const RationalSurface<T>& srf, T u, T v)
{

    //typedef glm::vec<4, T> tvecnp1;

    // Compute homogenous coordinates of control points
    array2<MPoint> Cw;
    Cw.resize(srf.control_points.rows(), srf.control_points.cols());
    for (int i = 0; i < srf.control_points.rows(); i++)
    {
        for (int j = 0; j < srf.control_points.cols(); j++)
        {
            Cw(i, j) = MPoint(cartesianToHomogenous(srf.control_points(i, j), srf.weights(i, j)));
        }
    }

    // Compute point using homogenous coordinates
    MPoint pointw = surfacePoint(srf.degree_u, srf.degree_v, srf.knots_u, srf.knots_v, Cw, u, v);

    // Convert back to cartesian coordinates
    return homogenousToCartesian(pointw);
}

/**
 * Evaluate derivatives on a non-rational NURBS surface
 * @param[in] degree_u Degree of the given surface in u-direction.
 * @param[in] degree_v Degree of the given surface in v-direction.
 * @param[in] knots_u Knot vector of the surface in u-direction.
 * @param[in] knots_v Knot vector of the surface in v-direction.
 * @param[in] control_points Control points of the surface in a 2D array.
 * @param[in] num_ders Number of times to differentiate
 * @param[in] u Parameter to evaluate the surface at.
 * @param[in] v Parameter to evaluate the surface at.
 * @return surf_ders Derivatives of the surface at (u, v).
 */
template <typename T>
array2<MVector> surfaceDerivatives(const Surface<T>& srf, int num_ders, T u, T v)
{
    return surfaceDerivatives(srf.degree_u, srf.degree_v, srf.knots_u, srf.knots_v,
        srf.control_points, num_ders, u, v);
}

/**
 * Evaluate derivatives on a rational NURBS surface
 * @param[in] srf RationalSurface object
 * @param[in] u Parameter to evaluate the surface at.
 * @param[in] v Parameter to evaluate the surface at.
 * @param[in] num_ders Number of times to differentiate
 * @return Derivatives on the surface at parameter (u, v).
 */
template <typename T>
array2<MVector> surfaceDerivatives(const RationalSurface<T>& srf, int num_ders, T u, T v)
{

    using namespace std;
    using namespace glm;

    /*typedef vec<3, T> tvecn;
    typedef vec<4, T> tvecnp1;*/

    array2<MPoint> homo_cp;
    homo_cp.resize(srf.control_points.rows(), srf.control_points.cols());
    for (int i = 0; i < srf.control_points.rows(); ++i)
    {
        for (int j = 0; j < srf.control_points.cols(); ++j)
        {
            homo_cp(i, j) = cartesianToHomogenous(srf.control_points(i, j), srf.weights(i, j));
        }
    }

    array2<MPoint> homo_ders = surfaceDerivatives(
        srf.degree_u, srf.degree_v, srf.knots_u, srf.knots_v, homo_cp, num_ders, u, v);

    array2<MVector> Aders;
    Aders.resize(num_ders + 1, num_ders + 1);
    for (int i = 0; i < homo_ders.rows(); ++i)
    {
        for (int j = 0; j < homo_ders.cols(); ++j)
        {
            Aders(i, j) = truncateHomogenous(homo_ders(i, j));
        }
    }

    array2<MVector> surf_ders(num_ders + 1, num_ders + 1);
    for (int k = 0; k < num_ders + 1; ++k)
    {
        for (int l = 0; l < num_ders - k + 1; ++l)
        {
            auto der = Aders(k, l);

            for (int j = 1; j < l + 1; ++j)
            {
                der -= (T)binomial(l, j) * homo_ders(0, j).w * surf_ders(k, l - j);
            }

            for (int i = 1; i < k + 1; ++i)
            {
                der -= (T)binomial(k, i) * homo_ders(i, 0).w * surf_ders(k - i, l);

                //tvecn tmp((T)0.0);
                MVector tmp();
                for (int j = 1; j < l + 1; ++j)
                {
                    tmp -= (T)binomial(l, j) * homo_ders(i, j).w * surf_ders(k - 1, l - j);
                }

                der -= (T)binomial(k, i) * tmp;
            }

            der *= 1 / homo_ders(0, 0).w;
            surf_ders(k, l) = der;
        }
    }
    return surf_ders;
}

/**
 * Evaluate the two orthogonal tangents of a non-rational surface at the given
 * parameters
 * @param[in] srf Surface object
 * @param u Parameter in the u-direction
 * @param v Parameter in the v-direction
 * @return Tuple with unit tangents along u- and v-directions
 */
template <typename T>
std::tuple<MVector, MVector> surfaceTangent(const Surface<T>& srf, T u, T v)
{
    array2<MVector> ptder = surfaceDerivatives(srf, 1, u, v);
    MVector du = ptder(1, 0);
    MVector dv = ptder(0, 1);
    /*T du_len = glm::length(ptder(1, 0));
    T dv_len = glm::length(ptder(0, 1));*/
    T du_len = ptder(1, 0).length();
    T dv_len = ptder(0, 1).length();
    if (!EQ(du_len, T(0)))
    {
        du /= du_len;
    }
    if (!EQ(dv_len, T(0)))
    {
        dv /= dv_len;
    }
    return std::make_tuple(std::move(du), std::move(dv));
}

/**
 * Evaluate the two orthogonal tangents of a rational surface at the given
 * parameters
 * @param[in] srf Rational Surface object
 * @param u Parameter in the u-direction
 * @param v Parameter in the v-direction
 * @return Tuple with unit tangents along u- and v-directions
 */
template <typename T>
std::tuple<MVector, MVector> surfaceTangent(const RationalSurface<T>& srf, T u, T v)
{
    array2<MVector> ptder = surfaceDerivatives(srf, 1, u, v);
    MVector du = ptder(1, 0);
    MVector dv = ptder(0, 1);
    /*T du_len = glm::length(ptder(1, 0));
    T dv_len = glm::length(ptder(0, 1));*/
    T du_len = ptder(1, 0).length();
    T dv_len = ptder(0, 1).length();
    if (!EQ(du_len, T(0)))
    {
        du /= du_len;
    }
    if (!EQ(dv_len, T(0)))
    {
        dv /= dv_len;
    }
    return std::make_tuple(std::move(du), std::move(dv));
}

/**
 * Evaluate the normal a non-rational surface at the given parameters
 * @param[in] srf Surface object
 * @param u Parameter in the u-direction
 * @param v Parameter in the v-direction
 * @param[inout] normal Unit normal at of the surface at (u, v)
 */
template <typename T> MVector surfaceNormal(const Surface<T>& srf, T u, T v)
{
    array2<MVector> ptder = surfaceDerivatives(srf, 1, u, v);
    //MVector n = glm::cross(ptder(0, 1), ptder(1, 0));
    MVector n = ptder(0, 1) ^ ptder(1, 0);
    //T n_len = glm::length(n);
    T n_len = n.length();
    if (!EQ(n_len, T(0)))
    {
        n /= n_len;
    }
    return n;
}

/**
 * Evaluate the normal of a rational surface at the given parameters
 * @param[in] srf Rational Surface object
 * @param u Parameter in the u-direction
 * @param v Parameter in the v-direction
 * @return Unit normal at of the surface at (u, v)
 */
template <typename T> MVector surfaceNormal(const RationalSurface<T>& srf, T u, T v)
{
    array2<MVector> ptder = surfaceDerivatives(srf, 1, u, v);
    //MVector n = glm::cross(ptder(0, 1), ptder(1, 0));
    MVector n = ptder(0, 1) ^ ptder(1, 0);
    //T n_len = glm::length(n);
    T n_len = n.length();
    if (!EQ(n_len, T(0)))
    {
        n /= n_len;
    }
    return n;
}



}





