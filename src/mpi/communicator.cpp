// communicator.cpp - Custom MPI wrappers (halo exchange placeholder)
#include <mpi.h>
#include <vector>
#include <stdexcept>

namespace demgpu {

void mpiCheck(int code, const char* msg) {
    if (code != MPI_SUCCESS) {
        throw std::runtime_error(msg);
    }
}

void haloExchangeFloat(std::vector<float>& leftRecv, std::vector<float>& rightRecv,
                        const std::vector<float>& leftSend, const std::vector<float>& rightSend,
                        int leftRank, int rightRank, MPI_Comm comm) {
    MPI_Request reqs[4];
    int idx = 0;
    if (leftRank >= 0) {
        mpiCheck(MPI_Irecv(leftRecv.data(), (int)leftRecv.size(), MPI_FLOAT, leftRank, 100, comm, &reqs[idx++]), "Irecv left failed");
        mpiCheck(MPI_Isend(leftSend.data(), (int)leftSend.size(), MPI_FLOAT, leftRank, 101, comm, &reqs[idx++]), "Isend left failed");
    }
    if (rightRank >= 0) {
        mpiCheck(MPI_Irecv(rightRecv.data(), (int)rightRecv.size(), MPI_FLOAT, rightRank, 101, comm, &reqs[idx++]), "Irecv right failed");
        mpiCheck(MPI_Isend(rightSend.data(), (int)rightSend.size(), MPI_FLOAT, rightRank, 100, comm, &reqs[idx++]), "Isend right failed");
    }
    if (idx > 0) {
        mpiCheck(MPI_Waitall(idx, reqs, MPI_STATUSES_IGNORE), "Waitall failed");
    }
}

} // namespace demgpu
