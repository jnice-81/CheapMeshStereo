/*
Reconstruction based on Poisson Surface reconstruction. Code in this file is mostly copied
from Reconstruction.example.cpp and follows there licence:

Copyright (c) 2023, Michael Kazhdan
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#pragma warning(push, 0)
#include <Reconstructors.h>
#pragma warning(pop)

template <typename Index, typename Real, unsigned int Dim>
class PoissonSurfaceReconstruct {
private:

	struct PolygonStream : public Reconstructor::OutputPolygonStream
	{
		// Construct a stream that adds polygons to the vector of polygons
		PolygonStream(std::vector< std::vector< Index > >& polygonStream) : _polygons(polygonStream) {}

		// Override the pure abstract method from OutputPolygonStream
		void base_write(const std::vector< node_index_type >& polygon)
		{
			std::vector< Index > poly(polygon.size());
			for (unsigned int i = 0; i < polygon.size(); i++) poly[i] = (Index)polygon[i];
			_polygons.push_back(poly);
		}
	protected:
		std::vector< std::vector< Index > >& _polygons;
	};

	struct VertexStream : public Reconstructor::OutputVertexStream< Real, Dim >
	{
		// Construct a stream that adds vertices into the coordinates
		VertexStream(std::vector< Real >& vCoordinates) : _vCoordinates(vCoordinates) {}

		// Override the pure abstract method from Reconstructor::OutputVertexStream< Real , Dim >
		void base_write(Point< Real, Dim > p, Point< Real, Dim >, Real) {
			for (unsigned int d = 0; d < Dim; d++) _vCoordinates.push_back(p[d]); 
		}
	protected:
		std::vector< Real >& _vCoordinates;
	};

	struct SceneInputStream : public Reconstructor::InputSampleStream<Real, Dim> {

		SceneInputStream(Scene &s) : scene(s) {
			current = s.getScenePoints().begin();
		}

		void reset() {
			current = scene.getScenePoints().begin();
		}

		bool base_read(Point<Real, Dim>& p, Point<Real, Dim>& n) {
			if (current != scene.getScenePoints().end()) {
				cv::Vec3f g = current->first;
				g = scene.addVoxelCenter(g * scene.getVoxelSideLength());
				cv::Vec3f w = current->second.normal;

				for (int i = 0; i < 3; i++) {
					p[i] = g[i];
					n[i] = w[i];
				}

				current++;
				return true;
			}
			
			return false;
		}

	private:
		std::unordered_map<cv::Vec3i, ScenePoint, VecHash>::iterator current;
		Scene& scene;
	};

public:
	static void reconstructSurface(Scene& scene,
		Reconstructor::Poisson::SolutionParameters<Real>& solverParams,
		Reconstructor::LevelSetExtractionParameters& extractionParams,
		std::vector<Real>& vCoordinates,
		std::vector<std::vector<Index>>& polygons,
		unsigned int num_threads = std::thread::hardware_concurrency()) {

		ThreadPool::Init(ThreadPool::THREAD_POOL, num_threads);

		typedef Reconstructor::Poisson ReconType;

		static const unsigned int FEMSig = FEMDegreeAndBType< ReconType::DefaultFEMDegree, ReconType::DefaultFEMBoundary >::Signature;
		SceneInputStream sceneInput(scene);
		PolygonStream polygonOutput(polygons);
		VertexStream vOutput(vCoordinates);

		ReconType::Implicit<Real, Dim, FEMSig> implicit(sceneInput, solverParams);
		implicit.extractLevelSet(vOutput, polygonOutput, extractionParams);

		ThreadPool::Terminate();
	}

	static void WritePly(std::string fileName, size_t vNum, const Real* vCoordinates, const std::vector< std::vector< Index > >& polygons)
	{
		std::fstream file(fileName, std::ios::out);
		file << "ply" << std::endl;
		file << "format ascii 1.0" << std::endl;
		file << "element vertex " << vNum << std::endl;
		file << "property float x" << std::endl << "property float y" << std::endl << "property float z" << std::endl;
		file << "element face " << polygons.size() << std::endl;
		file << "property list uchar int vertex_indices" << std::endl;
		file << "end_header" << std::endl;

		auto ColorChannel = [](Real v) { return std::max<int>(0, std::min<int>(255, (int)floor(255 * v + 0.5))); };

		for (size_t i = 0; i < vNum; i++)
		{
			file << vCoordinates[3 * i + 0] << " " << vCoordinates[3 * i + 1] << " " << vCoordinates[3 * i + 2];
			file << std::endl;
		}
		for (const auto& polygon : polygons)
		{
			file << polygon.size();
			for (auto vIdx : polygon) file << " " << vIdx;
			file << std::endl;
		}
	}

	static void reconstructAndExport(Scene &scene, std::string filename, int depth = 8) {
		Reconstructor::Poisson::SolutionParameters<Real> solverParams;
		solverParams.depth = depth;
		solverParams.verbose = false;
		Reconstructor::LevelSetExtractionParameters extractParams;
		extractParams.verbose = false;
		std::vector<Real> vertices;
		std::vector<std::vector<Index>> polygons;

		reconstructSurface(scene, solverParams, extractParams, vertices, polygons);
		WritePly(filename, vertices.size() / 3, vertices.data(), polygons);
	}
};