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

#include <catch2/catch.hpp>

#include <util/eigen_matnd.h>


TEMPLATE_TEST_CASE("Eigen/MatNd conversion", "[matrix]", Eigen::MatrixXd, Eigen::Matrix4d,
                   (Eigen::Matrix<double, 2, 3, Eigen::RowMajor>), Eigen::VectorXd, Eigen::RowVectorXd)
{
    int rows = TestType::RowsAtCompileTime;
    if (rows == -1) {
        // arbitrary dynamic value
        rows = 4;
    }
    int cols = TestType::ColsAtCompileTime;
    if (cols == -1) {
        // arbitrary dynamic value
        cols = 4;
    }
    
    // create random eigen matrix
    TestType eigen_mat = TestType::Random(rows, cols);
    
    // create MatNd
    MatNd* rcs_mat = NULL;
    MatNd_fromStack(rcs_mat, rows, cols)
    
    SECTION("Eigen to MatNd") {
        // perform copy
        Rcs::copyEigen2MatNd(rcs_mat, eigen_mat);
        
        // verify
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                DYNAMIC_SECTION("Entry [" << r << ", " << c << "]") {
                    CHECK(eigen_mat(r, c) == MatNd_get2(rcs_mat, r, c));
                }
            }
        }
    }
    
    SECTION("MatNd to Eigen") {
        TestType new_eigen_mat;
        
        // perform copy
        Rcs::copyMatNd2Eigen(new_eigen_mat, rcs_mat);
        
        // verify
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                DYNAMIC_SECTION("Entry [" << r << ", " << c << "]") {
                    CHECK(MatNd_get2(rcs_mat, r, c) == new_eigen_mat(r, c));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Wrapping Eigen as MatNd", "[matrix]", Eigen::VectorXd, Eigen::Vector4d, Eigen::RowVectorXd,
                   (Eigen::Matrix<double, 2, 3, Eigen::RowMajor>))
{
    int rows = TestType::RowsAtCompileTime;
    if (rows == -1) {
        // arbitrary dynamic value
        rows = 4;
    }
    int cols = TestType::ColsAtCompileTime;
    if (cols == -1) {
        // arbitrary dynamic value
        cols = 4;
    }
    
    // create random eigen matrix
    TestType eigen_mat = TestType::Random(rows, cols);
    
    // wrap
    MatNd rcs_mat = Rcs::viewEigen2MatNd(eigen_mat);
    
    // verify
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            DYNAMIC_SECTION("Entry [" << r << ", " << c << "]") {
                CHECK(eigen_mat(r, c) == MatNd_get2((&rcs_mat), r, c));
            }
        }
    }
}