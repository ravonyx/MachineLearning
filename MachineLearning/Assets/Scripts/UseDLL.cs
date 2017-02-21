using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System.IO;
using System.Collections;
using UnityEngine.UI;

public class UseDLL : MonoBehaviour
{
    [DllImport("MachineLearning", EntryPoint = "LinearCreateModel")]
    unsafe public static extern double* LinearCreateModel(int inputDimensions);

    [DllImport("MachineLearning", EntryPoint = "LinearFitClassificationClassic")]
    unsafe public static extern double* LinearFitClassificationClassic(double* model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize);

    [DllImport("MachineLearning", EntryPoint = "LinearClassify")]
    unsafe public static extern double* LinearClassify(double* model, double* inputs, int inputsSize);

    [DllImport("MachineLearning", EntryPoint = "LinearFitRegression")]
    unsafe public static extern double* LinearFitRegression(double* model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize);

    [DllImport("MachineLearning", EntryPoint = "LinearRegression")]
    unsafe public static extern double LinearRegression(double* model, double* inputs, int inputsSize);

    [SerializeField]
    Transform[] _referenceObjects;

    [SerializeField]
    Transform[] _fieldObjects;

    enum ParameterMode { None = 1, Square, Absolute, Test};
    ParameterMode paramMode = ParameterMode.None;

    // Update is called once per frame
    void Update()
    {
        #region Linear Classification
        if (Input.GetKeyDown(KeyCode.C))
        {
            double[] inputsRefCsharp = new double[_referenceObjects.Length * 2];
            double[] outputsRefCsharp = new double[_referenceObjects.Length];
            double[] inputsCsharp = new double[_fieldObjects.Length * 2];
            for (int i = 0; i < _referenceObjects.Length; i++)
            {
                if(paramMode == ParameterMode.None)
                {
                    inputsRefCsharp[i * 2] = _referenceObjects[i].position.x;
                    inputsRefCsharp[i * 2 + 1] = _referenceObjects[i].position.z;
                }
                else if (paramMode == ParameterMode.Square)
                {
                    inputsRefCsharp[i * 2] = Mathf.Pow(_referenceObjects[i].position.x,2);
                    inputsRefCsharp[i * 2 + 1] = Mathf.Pow(_referenceObjects[i].position.z,2);
                }
                else if (paramMode == ParameterMode.Absolute)
                {
                    inputsRefCsharp[i * 2] = Mathf.Abs(_referenceObjects[i].position.x);
                    inputsRefCsharp[i * 2 + 1] = Mathf.Abs(_referenceObjects[i].position.z);
                }
                else if (paramMode == ParameterMode.Absolute)
                {
                    inputsRefCsharp[i * 2] = Mathf.Abs(_referenceObjects[i].position.x);
                    inputsRefCsharp[i * 2 + 1] = Mathf.Abs(_referenceObjects[i].position.z);
                }
                outputsRefCsharp[i] = _referenceObjects[i].position.y > 0 ? 1 : 0;
            }

            for (int i = 0; i < _fieldObjects.Length; i++)
            {
                if (paramMode == ParameterMode.None)
                {
                    inputsCsharp[i * 2] = _fieldObjects[i].position.x;
                    inputsCsharp[i * 2 + 1] = _fieldObjects[i].position.z;
                }
                else if (paramMode == ParameterMode.Square)
                {
                    inputsCsharp[i * 2] = Mathf.Pow(_fieldObjects[i].position.x, 2);
                    inputsCsharp[i * 2 + 1] = Mathf.Pow(_fieldObjects[i].position.z, 2);

                }
                else if (paramMode == ParameterMode.Absolute)
                {
                    inputsCsharp[i * 2] = Mathf.Abs(_fieldObjects[i].position.x);
                    inputsCsharp[i * 2 + 1] = Mathf.Abs(_fieldObjects[i].position.z);
                }
                else if (paramMode == ParameterMode.Test)
                {
                    inputsCsharp[i * 2] = Mathf.Abs(_fieldObjects[i].position.x);
                    inputsCsharp[i * 2 + 1] = Mathf.Abs(_fieldObjects[i].position.z);
                }
            }

            unsafe
            {
                double* inputsRef = (double*)Marshal.UnsafeAddrOfPinnedArrayElement(inputsRefCsharp, 0);
                double* outputs = (double*)Marshal.UnsafeAddrOfPinnedArrayElement(outputsRefCsharp, 0);

                double* model = LinearCreateModel(2);

                LinearFitClassificationClassic(model, inputsRef, _referenceObjects.Length, 2, outputs, 1);

                for (int i = 0; i < _fieldObjects.Length; i++)
                {
                    double[] inputObject = new double[2];
                    inputObject[0] = inputsCsharp[i * 2];
                    inputObject[1] = inputsCsharp[i * 2 + 1];

                    double* inputs = (double*)Marshal.UnsafeAddrOfPinnedArrayElement(inputObject, 0);
                    _fieldObjects[i].position = new Vector3(_fieldObjects[i].position.x, (float)LinearClassify(model, inputs, 2)[0], _fieldObjects[i].position.z);
                }
            }
        }
        #endregion

        #region Linear Regression
        if (Input.GetKeyDown(KeyCode.R))
        {
            double[] inputsRefCsharp = new double[_referenceObjects.Length * 2];
            double[] outputsRefCsharp = new double[_referenceObjects.Length];
            double[] inputsCsharp = new double[_fieldObjects.Length * 2];

            for (int i = 0; i < _referenceObjects.Length; i++)
            {
                inputsRefCsharp[i * 2] = _referenceObjects[i].position.x;
                inputsRefCsharp[i * 2 + 1] = _referenceObjects[i].position.z;
                outputsRefCsharp[i] = _referenceObjects[i].position.y;
            }

            for (int i = 0; i < _fieldObjects.Length; i++)
            {
                inputsCsharp[i * 2] = _fieldObjects[i].position.x;
                inputsCsharp[i * 2 + 1] = _fieldObjects[i].position.z;
            }

            unsafe
            {
                double* inputsRef = (double*)Marshal.UnsafeAddrOfPinnedArrayElement(inputsRefCsharp, 0);
                double* outputs = (double*)Marshal.UnsafeAddrOfPinnedArrayElement(outputsRefCsharp, 0);

                double* model = LinearCreateModel(2);

                double* result = LinearFitRegression(model, inputsRef, _referenceObjects.Length, 2, outputs, 1);

                for (int i = 0; i < _fieldObjects.Length; i++)
                {
                    double[] inputObject = new double[2];
                    inputObject[0] = inputsCsharp[i * 2];
                    inputObject[1] = inputsCsharp[i * 2 + 1];

                    double* inputs = (double*)Marshal.UnsafeAddrOfPinnedArrayElement(inputObject, 0);
                    _fieldObjects[i].position = new Vector3(_fieldObjects[i].position.x, (float)LinearRegression(model, inputs, 2), _fieldObjects[i].position.z);
                }
            }
        }
        #endregion
    }

    public void onChangeParametersMode()
    {
        Debug.Log("On change parameter mode");
        Dropdown dropdown = GameObject.Find("Dropdown").GetComponent<Dropdown>();
        if (dropdown.value == 0)
            paramMode = ParameterMode.None;
        else if (dropdown.value == 1)
            paramMode = ParameterMode.Square;
        else if (dropdown.value == 2)
            paramMode = ParameterMode.Absolute;
        else if (dropdown.value == 3)
            paramMode = ParameterMode.Test;
    }
}
