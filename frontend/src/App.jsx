import { useState } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
    setResult(null)
    setError(null)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) throw new Error('API error')

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError("Failed to connect to backend. Is the FastAPI server running?")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <div style={{ padding: '40px', fontFamily: 'system-ui, sans-serif', maxWidth: '600px', margin: '0 auto' }}>
        <h1>Skin Cancer Detection AI</h1>
        <p>Upload a skin lesion image to get a prediction (currently using dummy model)</p>

        <form onSubmit={handleSubmit}>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'block', margin: '20px 0' }}
            required
          />
          <button
            type="submit"
            disabled={!file || loading}
            style={{
              padding: '12px 24px',
              fontSize: '16px',
              background: '#646cff',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer'
            }}
          >
            {loading ? 'Analyzing...' : 'Upload & Predict'}
          </button>
        </form>

        {error && <p style={{ color: 'red', marginTop: '20px' }}>{error}</p>}

        {result && (
          <div style={{ marginTop: '30px', padding: '20px', background: '#f0f0f0', borderRadius: '8px' }}>
            <h2>Result:</h2>
            <p><strong>Prediction:</strong> {result.prediction}</p>
            <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%</p>
            {result.note && <p><em>{result.note}</em></p>}
          </div>
        )}
      </div>
    </>
  )
}

export default App