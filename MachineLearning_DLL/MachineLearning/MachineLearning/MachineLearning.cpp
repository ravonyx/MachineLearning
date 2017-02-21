#include "MachineLearning.h"
#include <random>
#include <Eigen\Dense>
#include <iostream>
#include <Windows.h>

using namespace Eigen;

extern "C" 
{
	double appro = 0.1;

	double* LinearCreateModel(int inputDimensions)
	{
		double* model = new double[inputDimensions + 1];
		for (int i = 0; i < inputDimensions + 1; i++)
		{
			model[i] = rand() % 10 - 5;
		}

		return model;
	}

	double* LinearRemoveModel(double* model)
	{
		return nullptr;
	}

	double* LinearFitRegression(double* model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize)
	{
		MatrixXd X(inputsSize, inputSize + 1);
		MatrixXd Y(inputsSize, outputSize);
		
		for (int i = 0; i < inputsSize; i++)
		{
			X(i, 0) = 1;
			X(i, 1) = inputs[i * inputSize];
			X(i, 2) = inputs[i * inputSize + 1];
		}
		for (int i = 0; i < inputsSize; i++)
		{
			Y(i, 0) = outputs[i];
		}
		
		MatrixXd X1 = (X.transpose() * X);
		MatrixXd W = (X1.inverse() * X.transpose()) * Y;

		for (int i = 0; i <= inputSize; i++)
			model[i] = W(i, 0);

		return model;
	}

	// LinearFitClassificationRosenblatt
	double* LinearFitClassificationClassic(double* model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize)
	{
		int nbIter = 100;

		for (int k = 0; k < nbIter; k++)
		{
			for (int i = 0; i < inputsSize; i++)
			{
				double fitCalcul = model[0];
			
				for (int j = 0; j < inputSize; j++)
				{
					fitCalcul += model[j + 1] * inputs[i * inputSize + j];
				}

				if (fitCalcul != outputs[i])
				{
					int outPer = fitCalcul > 0 ? 1 : 0;
					model[0] += appro * (outputs[i] - outPer) * 1;

					for (int j = 0; j < inputSize; j++)
					{
						model[j + 1] += appro * (outputs[i] - outPer) *  inputs[i * inputSize + j];
					}
				}
			}
		}

		return model;
	}

	double* LinearFitClassificationHebb(double* model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize)
	{
		return nullptr;
	}
	
	double* LinearClassify(double* model, double* inputs, int inputsSize)
	{
		double fitCalcul = model[0];

		for (int i = 0; i < inputsSize; i++)
		{
			fitCalcul += model[i + 1] * inputs[i];
		}

		double* output = new double[1];
		if (fitCalcul >= 0)
		{
			output[0] = 1;
		}
		else
		{
			output[0] = -1;
		}
		return output;
	}

	double LinearRegression(double* model, double* inputs, int inputsSize)
	{
		MatrixXd X(1, inputsSize + 1);
		MatrixXd Y(1, 1);
		MatrixXd W(inputsSize + 1, 1);

		X(0, 0) = 1;
		for (int i = 0; i < inputsSize; i++)
		{
			X(0, i + 1) = inputs[i];
		}

		for (int i = 0; i <= inputsSize; i++)
		{
			W(i, 0) = model[i];
		}

		Y = X * W;

		return Y(0,0);
	}

	double* LinearPredict(double* model, double* inputs, int inputsSize)
	{
		return nullptr;
	}



	double*** LinearCreateMLPModel(int nbCouches, int* nbNeurones, int nbInputs)
	{
		double*** model = new double**[nbCouches];
		for (int i = 0; i < nbCouches; i++)
		{
			int nbNeuronesInLayer = nbNeurones[i];
			model[i] = new double*[nbNeuronesInLayer];
			for (int j = 0; j < nbNeuronesInLayer; j++)
			{
				model[i][j] = new double[nbInputs];
				for (int k = 0; k < nbInputs + 1; i++)
				{
					model[i][j][k] = rand() % 5 - 2;
				}
			}
		}

		return model;
	}


	double* LinearFitClassificationMulti(double*** model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize)
	{

		/*
		int nbIter = 100;

		for (int k = 0; k < nbIter; k++)
		{
			for (int i = 0; i < inputsSize; i++)
			{
				double fitCalcul = model[0];

				for (int j = 0; j < inputSize; j++)
				{
					fitCalcul += model[j + 1] * inputs[i * inputSize + j];
				}

				if (fitCalcul != outputs[i])
				{
					float outPer = tanh(fitCalcul);
					model[0] += appro * (outputs[i] - outPer) * 1;

					for (int j = 0; j < inputSize; j++)
					{
						model[j + 1] += appro * (outputs[i] - outPer) *  inputs[i * inputSize + j];
					}
				}
			}
		}

		return model;*/
	}

}