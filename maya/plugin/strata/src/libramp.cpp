
#include <algorithm>
#include <vector>
#include <maya/MIntArray.h>
#include <maya/MFloatArray.h>
#include <maya/MRampAttribute.h>

namespace strata {

	struct RampPoint {
		// pointers leading into maya float arrays
		float *position;
		float *value;
		int *interp;
		bool operator< (const RampPoint& other) const {
			return *position < *(other.position);
		}
	};

	struct RampInterface {
		/* 
		I don't think sorting actually matters much,
		the maya ramp interfaces don't need it
		*/

		//RampPoint* pointArray;
		std::vector<RampPoint> pointVec;
		unsigned int nPoints = 0;
		MIntArray indices, interps;
		MFloatArray positions, values;

		RampInterface(MRampAttribute& ramp) {
			if (nPoints < ramp.getNumEntries()) {
				nPoints = ramp.getNumEntries();
				pointVec.resize(nPoints);
				/*if (pointArray) delete[] pointArray;
				pointArray = new RampPoint[nPoints];*/
			}
			ramp.getEntries(indices, positions, values, interps);


			for (unsigned int i = 0; i < nPoints; i++)
			{
				pointVec[i].position = &positions[i];
				pointVec[i].value = &values[i];
				pointVec[i].interp = &interps[i];
			}
			//std::sort(pointArray, pointArray + nPoints);
			//sortPoints();
		}


		int appendPoints(int n) {
			// return the index of the first new point added - 
			// compare with struct.nPoints to find all added
			positions.setLength(n + nPoints);
			values.setLength(n + nPoints);
			interps.setLength(n + nPoints);
			pointVec.resize(n + nPoints);

			// relink all points
			for (unsigned int i = 0; i < nPoints + n; i++)
			{
				pointVec[i].position = &positions[i];
				pointVec[i].value = &values[i];
				pointVec[i].interp = &interps[i];
			}
			// update struct number of points
			nPoints = nPoints + n;
		}

		inline void sortPoints() {
			std::sort(pointVec.begin(), pointVec.end());
		}

		inline void getValueAtPosition(float position, float& value) {
			// stub for now, can be redone for better spline interpolation
		}

		inline void applyToRampAttribute(MRampAttribute& ramp) {
			ramp.setRamp(values, positions, interps);
		}

		~RampInterface() { // deallocate point array
		}
	};
}