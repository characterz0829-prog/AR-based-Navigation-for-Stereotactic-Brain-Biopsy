using UnityEngine;
using System.Collections.Generic;
using Accord.Math.Distances;
using MathNet.Numerics.LinearAlgebra;
using System.Linq;
using System;
using MathNet.Numerics.LinearAlgebra.Double;
using Accord.Collections;

public class HybridtRegistration : MonoBehaviour
{
    public GameObject targetObject;
    public Transform headModel;
    public Matrix<double> transformationMatrix;
    public List<Vector3> TargetPoints;
    public GameObject[] sourceObjects;


    public void StartRegistration()
    {
        TargetPoints = PointCloudManager.GetStablePoints();

        if (TargetPoints == null || TargetPoints.Count == 0)
        {
            Debug.Log("TargetPoints is null or empty.");
            return;
        }

        var SourceMatrix = GetWorldCoordinates(sourceObjects);
        var TargetMatrix = ConvertToMatrix(TargetPoints);

        var initPose = GetInitPoseFromHeadModel();
        transformationMatrix = TRICP(SourceMatrix, TargetMatrix, initPose);

        if (transformationMatrix == null || transformationMatrix.RowCount != 4 || transformationMatrix.ColumnCount != 4)
        {
            Debug.Log("Invalid transformation matrix.");
            return;
        }

        ApplyTransformToTarget();
    }

    Matrix<double> GetInitPoseFromHeadModel()
    {
        if (headModel == null)
        {
            Debug.LogError("HeadModel transform not assigned.");
            return DenseMatrix.CreateIdentity(4);
        }

        Matrix4x4 m = headModel.localToWorldMatrix;
        var T = DenseMatrix.CreateIdentity(4);

        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                T[r, c] = m[r, c];
            }
        }

        return T;
    }

    void ApplyTransformToTarget()
    {
        if (targetObject == null) return;

        Matrix4x4 m = new Matrix4x4();
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                m[r, c] = (float)transformationMatrix[r, c];

        Vector3 translation = m.GetColumn(3);
        Vector3 forward = m.GetColumn(2);
        Vector3 upwards = m.GetColumn(1);

        Quaternion rotation = Quaternion.LookRotation(forward, upwards);

        targetObject.transform.position += targetObject.transform.rotation * translation;
        targetObject.transform.rotation *= rotation;
    }

    Matrix<double> GetWorldCoordinates(GameObject[] objects)
    {
        var points = new double[objects.Length, 3];

        for (int i = 0; i < objects.Length; i++)
        {
            Vector3 p = objects[i].transform.position;
            points[i, 0] = p.x;
            points[i, 1] = p.y;
            points[i, 2] = p.z;
        }

        return Matrix<double>.Build.DenseOfArray(points);
    }

    public static Matrix<double> TRICP(Matrix<double> A, Matrix<double> B, Matrix<double> initPose = null, int maxIterations = 1000, double tolerance = 1e-6)
    {
        int m = A.ColumnCount;
        int nA = A.RowCount;
        int nB = B.RowCount;

        var src = DenseMatrix.Create(m + 1, nA, 1.0);
        var dst = DenseMatrix.Create(m + 1, nB, 1.0);

        src.SetSubMatrix(0, m, 0, nA, A.Transpose());
        dst.SetSubMatrix(0, m, 0, nB, B.Transpose());

        if (initPose != null)
        {
            src = (DenseMatrix)initPose * src;
        }

        double prevError = 0;

        for (int i = 0; i < maxIterations; i++)
        {
            var (distances, indices) =
                NearestNeighbor(
                    src.SubMatrix(0, m, 0, src.ColumnCount).Transpose(),
                    dst.SubMatrix(0, m, 0, dst.ColumnCount).Transpose());

            var T = BestFitTransform(
                src.SubMatrix(0, m, 0, indices.Length).Transpose(),
                dst.SubMatrix(0, m, 0, indices.Length).Transpose());

            src = (DenseMatrix)T * src;

            double meanError = distances.Average();
            if (Math.Abs(prevError - meanError) < tolerance)
                break;

            prevError = meanError;
        }

        var finalT = BestFitTransform(A, src.SubMatrix(0, m, 0, A.RowCount).Transpose());
        return finalT;
    }

    public static (double[], int[]) NearestNeighbor(Matrix<double> sourcePoints, Matrix<double> targetPoints)
    {
        var src = sourcePoints.ToRowArrays();
        var dst = targetPoints.ToRowArrays();

        var knn = new KDTree<int>(dst[0].Length);
        for (int i = 0; i < dst.Length; i++)
            knn.Add(dst[i], i);

        double[] distances = new double[src.Length];
        int[] indices = new int[src.Length];
        var euclidean = new Euclidean();

        for (int i = 0; i < src.Length; i++)
        {
            var nearest = knn.Nearest(src[i]);
            indices[i] = nearest.Value;
            distances[i] = euclidean.Distance(src[i], nearest.Position);
        }

        return (distances, indices);
    }

    public static Matrix<double> BestFitTransform(Matrix<double> A, Matrix<double> B)
    {
        var centroidA = A.ColumnSums() / A.RowCount;
        var centroidB = B.ColumnSums() / B.RowCount;

        var AA = A - DenseMatrix.OfRowVectors(Enumerable.Repeat(centroidA, A.RowCount));
        var BB = B - DenseMatrix.OfRowVectors(Enumerable.Repeat(centroidB, B.RowCount));

        var H = AA.TransposeThisAndMultiply(BB);
        var svd = H.Svd(true);

        var R = svd.VT.TransposeThisAndMultiply(svd.U.Transpose());
        if (R.Determinant() < 0)
        {
            svd.VT.SetRow(svd.VT.RowCount - 1, svd.VT.Row(svd.VT.RowCount - 1).Negate());
            R = svd.VT.TransposeThisAndMultiply(svd.U.Transpose());
        }

        var t = centroidB - R * centroidA;

        var T = DenseMatrix.CreateIdentity(4);
        T.SetSubMatrix(0, 3, 0, 3, R);
        T.SetSubMatrix(0, 3, 3, 1, t.ToColumnMatrix());

        return T;
    }

    public static Matrix<double> ConvertToMatrix(List<Vector3> points)
    {
        var matrix = Matrix<double>.Build.Dense(points.Count, 3);

        for (int i = 0; i < points.Count; i++)
        {
            matrix[i, 0] = points[i].x;
            matrix[i, 1] = points[i].y;
            matrix[i, 2] = points[i].z;
        }

        return matrix;
    }
}
