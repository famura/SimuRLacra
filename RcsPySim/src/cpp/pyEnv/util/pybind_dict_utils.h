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

#ifndef PYBIND_DICT_UTILS_H_
#define PYBIND_DICT_UTILS_H_

#include <pybind11/pybind11.h>

#include <dictobject.h>

namespace py = pybind11;

// Some utilities for quickly accessing dict elements. Mainly intended for use with kwargs, but also useful otherwise
// these take a py::dict reference to ensure that the object is properly cast, but save reference counts

/**
 * Get the string-labeled item from the dict and return it as python object.
 * Returns default_ if item not found.
 *
 * This is designed to work similar to python's dict.get()
 *
 * @param dict python dictionary
 * @param key key string
 * @param default_ default value to return if not found. defaults to None.
 * @param noneIsEmpty set to false to return None values explicitly. If true, None is replaced with default_.
 * @return the value for key if found, else default.
 */
inline py::object get(const py::dict& dict, const char* key, py::handle default_ = py::none(), bool noneIsEmpty = true)
{
    PyObject* result = PyDict_GetItemString(dict.ptr(), key);
    if (result && (!noneIsEmpty || result != Py_None)) {
        return py::reinterpret_borrow<py::object>(result);
    }
    else {
        PyErr_Clear();
        return py::reinterpret_borrow<py::object>(default_);
    }
}

/**
 * Get the string-labeled item from the dict and return it as converted c++ value.
 * Returns default_ if item not found.
 *
 * This is designed to work similar to python's dict.get().
 *
 * A special overload of this function handles pointer types, which are normally not allowed as cast return type.
 *
 * @param dict python dictionary
 * @param key key string
 * @param default_ default value to return if not found.
 * @param noneIsEmpty set to false to return None values explicitly. If true, None is replaced with default_.
 * @return the value for key if found, else default.
 */
template<typename T, typename std::enable_if<!std::is_pointer<T>::value, int>::type = 0>
inline T get_cast(const py::dict& dict, const char* key, T default_, bool noneIsEmpty = true)
{
    PyObject* result = PyDict_GetItemString(dict.ptr(), key);
    if (result && (!noneIsEmpty || result != Py_None)) {
        return py::handle(result).cast<T>();
    }
    else {
        PyErr_Clear();
        return std::forward<T>(default_);
    }
}

// special case for returning pointers, which allows default to be NULL.
// here, we need to return something storing the type caster, since the regular cast doesn't support pointers
namespace detail
{

// this object holds the type caster, and otherwise behaves as if it were a T*
template<typename T>
struct cast_pointer_proxy
{
private:
    py::detail::make_caster<T> caster;
    T* ptr;
public:
    cast_pointer_proxy(T* val)
    {
        ptr = val;
    }
    
    cast_pointer_proxy(const py::handle& handle)
    {
        if (handle.is_none()) {
            // some type casters don't support none
            ptr = NULL;
        }
        else {
            py::detail::load_type(caster, handle);
            ptr = py::detail::cast_op<T*>(caster);
        }
    }
    
    operator T*()
    {
        return ptr;
    }
    
    T* operator->()
    {
        return ptr;
    }
    
};

}


/**
 * Get the string-labeled item from the dict and return it as converted c++ pointer value value.
 * Returns default_ if item not found.
 *
 * This is designed to work similar to python's dict.get().
 *
 * The return value is actually a wrapper to make sure the pointer's target does not go out of scope.
 * It has operator overloads to be usable like a regular pointer.
 *
 * @param dict python dictionary
 * @param key key string
 * @param default_ default value to return if not found. defaults to NULL.
 * @return the value for key if found, else default.
 */
template<typename T, typename std::enable_if<std::is_pointer<T>::value, int>::type = 0>
inline detail::cast_pointer_proxy<typename std::remove_pointer<T>::type>
get_cast(const py::dict& dict, const char* key, T&& default_ = NULL)
{
    if (PyObject* result = PyDict_GetItemString(dict.ptr(), key)) {
        return {result};
    }
    else {
        PyErr_Clear();
        return {default_};
    }
}


/**
 * Get the string-labeled item from the dict, convert it and set it to the var parameter
 * Returns false if item not found.
 *
 * This is designed to work similar to python's dict.get().
 *
 * @param dict python dictionary
 * @param key key string
 * @param[out] var output var ref. not modified if the parameter is not found.
 * @return true if found
 */
template<typename T>
inline bool try_get(const py::dict& dict, const char* key, T& var)
{
    if (PyObject* result = PyDict_GetItemString(dict.ptr(), key)) {
        var = pybind11::handle(result).cast<T>();
        return true;
    }
    else {
        PyErr_Clear();
        return false;
    }
}

/**
 * Get the string-labeled item from the dict, convert it and set it to the var parameter
 * Returns false if item not found.
 *
 * This is designed to work similar to python's dict.get().
 *
 * @param dict python dictionary
 * @param key key string
 * @param[out] var output var ref. not modified if the parameter is not found.
 * @return true if found
 */
template<typename T>
inline bool try_get(const py::dict& dict, const std::string& key, T& var)
{
    if (PyObject* result = PyDict_GetItemString(dict.ptr(), key.c_str())) {
        var = pybind11::handle(result).cast<T>();
        return true;
    }
    else {
        PyErr_Clear();
        return false;
    }
}


#endif /* PYBIND_DICT_UTILS_H_ */
