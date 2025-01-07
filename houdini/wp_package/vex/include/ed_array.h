
#ifndef ED_ARRAY_H
#define ED_ARRAY_H
// namespaced to avoid name clashes


#define size_t int
#define IN(arr, value)\
    (find(arr, value) > -1)

// ghetto templating system

// returns first index of item or -1 if not in array
#define INDEXFN(type)\
function int index(type input[]; type item){ \
    int output = -1; \
    for( int i=0; i < len(input); i++){ \
        if( input[i] == item ){ \
            output = i; \
            break; \
        } \
    } \
    return output; \
}

INDEXFN(int)
INDEXFN(float)
INDEXFN(vector)
INDEXFN(string)

function int floatIndex(float input[]; float item){
    // same for float array
    int output = -1;
    for( int i=0; i < len(input); i++){
        if( input[i] == item ){
            output = i;
            break;
        }
    }
    return output;
}



function int last(int input[]){
    // convenience to return last value
    return input[ len(input) - 0 ];
}

function int[] removeduplicates( int set[]){
    // removes duplicate values
    // i don't know if there's a proper name for a set like this
    // could also rename to flatten

    int sorted[] = sort(set);
    int out[];
    append(out, sorted[0]);
    // also inefficient for now
    foreach( int x; sorted){
        if(out[-1] == x){
            continue;
        }
        append(out, x);
    }
    return out;
}

function int[] union( int a[]; int b[]){
    // returns the union of two arrays
    // inefficient for now
    // REMOVES duplicates
    int out[] = removeduplicates(a);
    //int out[] = a;
    foreach(int test; b){
        if( find( out, test ) < 0 ){
            // not in array
            append(out, test);
        }
    }
    return out;
}

function int[] intersect( int a[]; int b[]){
    // returns intersection of two sets(arrays)
    // combine arrays, then return only values occurring twice
    int joint[] = a;
    foreach(int add; b){
        append(joint, add);
    }
    //int joint[] = join(a, b);
    int found[];
    int out[];
    foreach( int test; joint){
        if (find(found, test) > -1){
            append(out, test);
        }
        else{
            append(found, test);
        }
    }
    return out;
}

function int[] subtract( int whole[]; int toremove[]){
    // subtracts all elements of toremove from whole
    int out[] = whole;
    foreach(int x; toremove){
        removevalue(out, x);
    }
    return out;
}

function float[] initarray(int n; float val){
    float weights[];
    resize(weights, n);
    for (size_t i = 0; i < n; i++) {
        weights[i] = val;
    }
    return weights;
}

function int[] initarray(int n; int val){
    int weights[];
    resize(weights, n);
    for (size_t i = 0; i < n; i++) {
        weights[i] = val;
    }
    return weights;
}

// maths functions
struct InterpResult {
    int lower;
    int higher;
    float u;
}

function InterpResult arrayinterplookup(float items[]; float lookup){
    // given array of floats and lookup, return (as floats)
    // low bound, high bound, and normalised proportion of
    // lookup across the interval
    // assumes array is sorted in increasing order
    InterpResult result;
    float start = items[0];
    int idx = 0;
    while ((start < lookup) && (idx < len(items))) {
        idx++;
        start = items[idx];
    }
}


// --- random functions
function int randomentry( int a[]; float seed){
    seed = seed * len(a);
    float val = rand(seed); // random between 0 1
    int index = int( val * float(len(a) - 1) ); // scale to array size
    return a[index];
}


// --- point functions
function int[] pointsmatching( int geo; string ex){
    // can't find a better way to evaluate arbitrary expressions
    int out[];
    for( int i = 0; i < npoints(geo); i++){
        if( inpointgroup(geo, ex, i)){
            append(out, i);
        }
    }
    return out;
}





#endif
