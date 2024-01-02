'use client'

import React, { useState } from 'react'
import { CardHeader, CardContent, Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

type ApiResponse = {
  accuracy: number
  confusion_matrix: number[][]
  evaluation_time: number
  file: string
  number_of_attributes: number
  number_of_classes: number
  number_of_examples: number
  training_time: number
}

export function NaiveBayes() {
  const [dataset, setDataset] = useState('iris') // default to iris
  const [validationType, setValidationType] = useState('standard') // default to standard
  const [results, setResults] = useState<ApiResponse | null>(null)

  const handleDatasetClick = (selectedDataset: string) => {
    setDataset(selectedDataset)
  }

  const handleValidationTypeClick = (type: string) => {
    setValidationType(type)
  }

  const handlePredictClick = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dataset: dataset,
          prediction_type: validationType,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`)
      }

      const data: ApiResponse = await response.json()
      setResults(data)
    } catch (error) {
      console.error('Error during API call', error)
    }
  }

  const getButtonClass = (buttonType: string, currentSelection: string) => {
    const baseClass = 'px-4 py-2 rounded shadow mx-2 text-white '
    return currentSelection === buttonType
      ? baseClass + 'bg-primary/90' // Highlighted color
      : baseClass + 'bg-blue-500' // Normal color
  }

  const renderConfusionMatrix = (matrix: number[][]) => {
    return (
      <table className='min-w-full divide-y divide-gray-200 mt-2'>
        <thead>
          <tr className='bg-gray-50'>
            <th className='px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>
              
            </th>
            {matrix[0].map((_, idx) => (
              <th key={idx} className='px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>
                {idx}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className='bg-white divide-y divide-gray-200'>
          {matrix.map((row, idx) => (
            <tr key={idx}>
              <td className='px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900'>{idx}</td>
              {row.map((cell, cellIdx) => (
                <td key={cellIdx} className='px-6 py-4 whitespace-nowrap text-sm text-gray-500'>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  return (
    <div className='flex justify-center items-center min-h-screen bg-gray-200'>
      <Card className='m-4 p-4 bg-white rounded shadow-md'>
        <CardHeader className='flex items-center justify-between'>
          <h2 className='text-xl font-bold text-black'>Naive Bayes Predictions</h2>
        </CardHeader>
        <CardContent className='grid grid-cols-2 gap-4'>
          <div className='col-span-2 mb-4'>
            <div className='flex justify-center mt-4'>
              <Button
                className={getButtonClass('iris', dataset)}
                onClick={() => handleDatasetClick('iris')}
              >
                Iris Dataset
              </Button>
              <Button
                className={getButtonClass('banknote', dataset)}
                onClick={() => handleDatasetClick('banknote')}
              >
                Banknote Dataset
              </Button>
            </div>
            <div className='flex justify-center mt-4'>
              <Button
                className={getButtonClass('standard', validationType)}
                onClick={() => handleValidationTypeClick('standard')}
              >
                Standard Validation
              </Button>
              <Button
                className={getButtonClass('crossval', validationType)}
                onClick={() => handleValidationTypeClick('crossval')}
              >
                5-Fold Cross Validation
              </Button>
            </div>
            <div className='flex justify-center mt-4'>
              <Button
                className='bg-green-500 text-white px-4 py-2  shadow mx-2'
                onClick={handlePredictClick}
              >
                Predict
              </Button>
            </div>
          </div>
          {results && (
            <>
              <div>
                <h3 className='text-lg font-medium text-gray-700 mb-2'>Performance</h3>
                <p className='text-gray-500'>
                  Accuracy:{' '}
                  <span className='font-bold text-black'>
                    {(results.accuracy * 100).toFixed(2)}%
                  </span>
                </p>
                <p className='text-gray-500'>
                  Evaluation Time:{' '}
                  <span className='font-bold text-black'>
                    {results.evaluation_time.toFixed(3)} seconds
                  </span>
                </p>
                <p className='text-gray-500'>
                  Training Time:{' '}
                  <span className='font-bold text-black'>
                    {results.training_time.toFixed(3)} seconds
                  </span>
                </p>
              </div>

              <div className='flex flex-col items-center justify-center'>
                <h3 className='text-lg font-medium text-gray-700 mb-2'>Confusion Matrix</h3>
                {renderConfusionMatrix(results.confusion_matrix)}
              </div>

              <div>
                <h3 className='text-lg font-medium text-gray-700 mb-2'>Dataset</h3>
                <p className='text-gray-500'>
                  File: <span className='font-bold text-black'>{results.file}</span>
                </p>
                <p className='text-gray-500'>
                  Number of Attributes:{' '}
                  <span className='font-bold text-black'>{results.number_of_attributes}</span>
                </p>
                <p className='text-gray-500'>
                  Number of Classes:{' '}
                  <span className='font-bold text-black'>{results.number_of_classes}</span>
                </p>
                <p className='text-gray-500'>
                  Number of Examples:{' '}
                  <span className='font-bold text-black'>{results.number_of_examples}</span>
                </p>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
