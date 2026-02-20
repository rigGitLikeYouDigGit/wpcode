#pragma once

#include <maya/MPxObjectSet.h>
#include <maya/MObject.h>
#include <maya/MString.h>
#include <maya/MTypeId.h>
#include <maya/MCallbackIdArray.h>

/**
 * Maya object set plugin node that manages membership based on a string expression.
 * Whenever a new node is added to the scene, this set checks if it matches the 
 * expression (similar to cmds.ls()) and automatically adds it to the set.
 */

namespace wp {
    class ExpSet : public MPxObjectSet {
    public:
        ExpSet();
        ~ExpSet() override;

        // MPxNode overrides
        void postConstructor() override;

        // Node registration
        static void* creator();
        static MStatus initialize();

        // Node type info
        static const MString kNODE_NAME;
        static const MTypeId kNODE_ID;
        static const MString typeName;

        // Attributes
        static MObject aExpression;      // String expression for matching nodes
        static MObject aAutoUpdate;      // Enable/disable auto-updating
        static MObject aBalanceWheel;    // Trigger attribute for updates

    private:
        // Update set membership based on current expression
        MStatus updateMembership();

        // Callback for when nodes are added to the scene
        static void onNodeAdded(MObject& node, void* clientData);

        // Remove all callbacks
        void removeCallbacks();

        // Callback IDs for cleanup
        MCallbackIdArray m_callbackIds;

        // Flag to prevent recursive updates
        bool m_isUpdating;
    };
}