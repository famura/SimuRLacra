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

#ifndef RCSPYSIM_EIGEN_MATND_H
#define RCSPYSIM_EIGEN_MATND_H

#include <Rcs_MatNd.h>

#include <Eigen/Core>

namespace Rcs
{

/**
 * View an eigen matrix as Rcs MatNd.
 * @tparam M eigen matrix type
 * @param src eigen matrix
 * @return MatNd struct referencing the data of src.
 */
template<typename M>
MatNd viewEigen2MatNd(M& src)
{
    static_assert(std::is_same<typename M::Scalar, double>::value,
                  "MatNd is only compatible with double matrices.");
    static_assert(M::IsRowMajor || M::IsVectorAtCompileTime,
                  "MatNd is only compatible with row major matrices or vectors.");
    MatNd mat;
    mat.m = src.rows();
    mat.n = src.cols();
    mat.size = src.size();
    mat.ele = src.data();
    mat.stackMem = true;
    return mat;
}

// MatNd is row major. Eigen, by default, is column major.
// This alias provides a type closest to M that has a MatNd-compatible layout.
// For vectors, we do nothing, since there are no inner strides.
template<typename M>
using MatNdMappable = typename std::conditional<
    M::IsRowMajor || M::IsVectorAtCompileTime,
    M,
    Eigen::Matrix<
        typename M::Scalar,
        M::RowsAtCompileTime,
        M::ColsAtCompileTime,
        Eigen::RowMajor> >::type;


/**
 * Wrap a Rcs MatNd in a mutable Eigen Map object.
 *
 * If M is column major, the resulting map will still wrap a row major matrix, since that is the storage
 * order of MatNd.
 *
 * @tparam M desired eigen matrix type
 * @param src rcs matrix
 * @return Eigen::Map referencing the data of src.
 */
template<typename M = Eigen::MatrixXd>
Eigen::Map<MatNdMappable<M>> viewMatNd2Eigen(MatNd* src)
{
    static_assert(std::is_same<typename M::Scalar, double>::value,
                  "MatNd is only compatible with double matrices.");
    return Eigen::Map<MatNdMappable<M>>(src->ele, src->m, src->n);
}

/**
 * Wrap a read-only Rcs MatNd in a const Eigen Map object.
 *
 * If M is column major, the resulting map will still wrap a row major matrix, since that is the storage
 * order of MatNd.
 *
 * @tparam M desired eigen matrix type
 * @param src rcs matrix
 * @return Eigen::Map referencing the data of src.
 */
template<typename M = Eigen::MatrixXd>
Eigen::Map<const MatNdMappable<M>> viewMatNd2Eigen(const MatNd* src)
{
    static_assert(std::is_same<typename M::Scalar, double>::value,
                  "MatNd is only compatible with double matrices.");
    return Eigen::Map<const MatNdMappable<M>>(src->ele, src->m, src->n);
}

/**
 * Copy a matrix from Rcs to Eigen.
 * @tparam M eigen matrix type
 * @param dst destination eigen matrix
 * @param src source rcs matrix
 */
template<typename M>
void copyMatNd2Eigen(M& dst, const MatNd* src)
{
    dst = viewMatNd2Eigen<M>(src);
}

/**
 * Copy a matrix from Eigen to Rcs.
 * @tparam M eigen matrix type
 * @param dst destination rcs matrix
 * @param src source eigen matrix
 */
template<typename M>
void copyEigen2MatNd(MatNd* dst, const M& src)
{
    viewMatNd2Eigen<M>(dst) = src;
}

}

#endif //RCSPYSIM_EIGEN_MATND_H
