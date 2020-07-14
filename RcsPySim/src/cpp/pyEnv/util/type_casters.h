/*******************************************************************************
 Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
 Technical University of Darmstadt.
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
    or Technical University of Darmstadt, nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
 OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

#ifndef SRC_TYPE_CASTERS_H_
#define SRC_TYPE_CASTERS_H_

#include <pybind11/cast.h>
#include <pybind11/numpy.h>

#include <Rcs_MatNd.h>

/*
 * pybind11 type caster for RCS MatNd.
 *
 * A lot of this is adapted from the Eigen typecasters coming with pybind11.
 */
namespace pybind11
{
namespace detail
{

// pybind11 numpy array type with the same shape as the MatNd storage
typedef pybind11::array_t<double, pybind11::array::forcecast | pybind11::array::c_style> ArrayMatNd;

/*
 * Populate a MatNd so that it views the given numpy array's storage.
 */
static MatNd MatNd_fromNumpy(ArrayMatNd& nparray)
{
    // load dimensions
    unsigned int m = 1;
    unsigned int n = 1;
    if (nparray.ndim() >= 1) {
        m = nparray.shape(0);
    }
    if (nparray.ndim() == 2) {
        n = nparray.shape(1);
    }
    return MatNd_fromPtr(m, n, const_cast<double*>(nparray.data()));
}

/*
 * Create a numpy array from the given MatNd.
 *
 * If parent is NULL (ie default initialized handle), the data will be copied
 * If parent is None (python), an unguarded reference will be held
 * If parent is any other object, it is assumed that that object own mat.
 */
static handle MatNd_toNumpy(const MatNd* mat, handle parent = handle(), bool writeable = true)
{
    // create with dimensions, optimizing if mat->n = 1
    ArrayMatNd array;
    if (mat->n == 1) {
        // just a vector
        array = ArrayMatNd(mat->m, mat->ele, parent);
    }
    else {
        // a regular matrix
        array = ArrayMatNd({mat->m, mat->n}, mat->ele, parent);
    }
    
    // set read-only if needed
    if (!writeable) {
        array_proxy(array.ptr())->flags &= ~detail::npy_api::NPY_ARRAY_WRITEABLE_;
    }
    
    // release the handle to keep up the reference count!
    return array.release();
}

/*
 * Create a numpy array holding a reference to the given MatNd
 */
static handle MatNd_toNumpyRef(const MatNd* mat, handle parent = none())
{
    return MatNd_toNumpy(mat, parent, false);
}

/*
 * Create a numpy array holding a reference to the given MatNd
 */
static handle MatNd_toNumpyRef(MatNd* mat, handle parent = none())
{
    return MatNd_toNumpy(mat, parent, true);
}

/*
 * Create a numpy array taking ownership of the given MatNd
 */
static handle MatNd_toNumpyOwn(MatNd* mat)
{
    capsule caps(mat, [](void* ref) { MatNd_destroy((MatNd*) ref); });
    return MatNd_toNumpy(mat, caps);
}


template<>
struct type_caster<MatNd>
{
public:
    /**
     * Function signature for documentation
     */
#if PYBIND11_VERSION_MINOR >= 3
    static constexpr auto name = _("MatNd");
#else
    static PYBIND11_DESCR name() { return type_descr(_("MatNd")); }
#endif


protected:
    // local value
    MatNd value;
    
    // a reference to the numpy array backing value
    ArrayMatNd copyOrRef;
    
    // a double type caster used to process scalars as 1x1 matrices
    make_caster<double> scalar_caster;
    
    // true if we're None a.k.a. NULL
    bool none = false;

public:
    /**
     * Access to the internal loaded value store.
     */
    operator MatNd*()
    {
        if (none) {
            return NULL;
        }
        return &value;
    }
    
    operator MatNd&()
    {
        if (none) {
            throw value_error("Cannot convert None to a MatNd ref");
        }
        
        return value;
    }
    
    template<typename T_> using cast_op_type = pybind11::detail::cast_op_type<T_>;
    
    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a MatNd
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!convert && !isinstance<ArrayMatNd>(src)) {
            return false;
        }
        if (src.is_none()) {
            // allow none and pass it as NULL
            none = true;
            return true;
        }
        
        if (scalar_caster.load(src, convert)) {
            // scalars should be handled as 1x1 mats. ArrayMatNd::ensure is capable of this,
            // but would allocate memory which is unnecessary.
            value.ele = cast_op<double*>(scalar_caster);
            value.m = 1;
            value.n = 1;
            value.size = 1;
            value.stackMem = true;
            return true;
        }
        
        // ensure and cast if needed. This copies only if needed
        copyOrRef = ArrayMatNd::ensure(src);
        
        // we can only handle arrays of 2 or less dimensions
        auto dims = copyOrRef.ndim();
        if (dims > 2) {
            return false;
        }
        
        // fill value so that it uses the same memory as copyOrRef
        value = MatNd_fromNumpy(copyOrRef);
        return true;
    }
    
    /**
     * Conversion part 2 (C++ -> Python): convert a MatNd instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``).
     */
    static handle cast(MatNd* src, return_value_policy policy, handle parent)
    {
        switch (policy) {
            case return_value_policy::take_ownership:
            case return_value_policy::automatic:
                // this is not entirely correct, since usually we wouldn't support this for pointers at all.
                // however, when used in a tuple, we will always get this policy, so we need to deal with it.
                // and after a fashion, moving a pointer is essentially the same as taking ownership
            case return_value_policy::move:
                
                return MatNd_toNumpyOwn(src);
            case return_value_policy::copy:
                return MatNd_toNumpy(src);
            case return_value_policy::reference:
            case return_value_policy::automatic_reference:
                return MatNd_toNumpyRef(src);
            case return_value_policy::reference_internal:
                return MatNd_toNumpyRef(src, parent);
            default:
                pybind11_fail("Invalid return_value_policy for Rcs MatNd type");
        };
    }
    
    static handle cast(const MatNd* src, return_value_policy policy, handle parent)
    {
        switch (policy) {
            case return_value_policy::copy:
                return MatNd_toNumpy(src);
            case return_value_policy::reference:
            case return_value_policy::automatic_reference:
                return MatNd_toNumpyRef(src);
            case return_value_policy::reference_internal:
                return MatNd_toNumpyRef(src, parent);
            default:
                // can also not take ownership of a const matrix
                pybind11_fail("Invalid return_value_policy for Rcs MatNd type");
        };
    }
};

}
} // namespace pybind11::detail


#endif /* SRC_TYPE_CASTERS_H_ */
