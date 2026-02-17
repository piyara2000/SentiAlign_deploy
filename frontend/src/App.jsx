import { useState } from 'react'
import { Routes, Route, useLocation, useNavigate, Navigate } from 'react-router-dom'
import './App.css'

function percent(x) {
  if (typeof x !== 'number' || Number.isNaN(x)) return '-'
  return `${(x * 100).toFixed(2)}%`
}

function AnalyzePage() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [includeShap, setIncludeShap] = useState(false)
  const navigate = useNavigate()

  async function analyze() {
    const trimmed = text.trim()
    if (!trimmed) {
      setError('Please enter some text to analyze.')
      return
    }
    setError('')
    setLoading(true)
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: trimmed, include_shap: includeShap }),
      })

      const data = await res.json()
      if (!res.ok || data?.error) {
        setError(data?.error || 'Analyze failed.')
        return
      }

      // Navigate to /results with data in router state
      navigate('/results', {
        state: {
          text: trimmed,
          results: data,
        },
      })
    } catch (e) {
      console.error(e)
      setError('An error occurred while analyzing the text. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header>
        <h1 className="title">SentiAlign</h1>
        <p className="subtitle">
          Analyze text sentiment using multiple AI models with explainable results
        </p>
      </header>

      <main>
        <div className="card">
          <label htmlFor="text-input" className="input-label">
            Insert Text to Analyze:
          </label>
          <textarea
            id="text-input"
            className="text-input"
            placeholder="Type or Paste your text here ..."
            rows={8}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyze()
            }}
          />

          <label className="shap-toggle">
            <input
              type="checkbox"
              checked={includeShap}
              onChange={(e) => setIncludeShap(e.target.checked)}
            />
            <span>
              Show detailed feature contributions (SHAP)
            </span>
          </label>

          <button className="btn-primary" onClick={analyze} disabled={loading}>
            <span>{loading ? 'Analyzing…' : 'Analyze'}</span>
            <span className="arrow">→</span>
          </button>
        </div>

        {loading ? (
          <div className="loading">Analyzing sentiment...</div>
        ) : null}

        {error ? <div className="error">{error}</div> : null}
      </main>
    </div>
  )
}

function ResultsPage() {
  const location = useLocation()
  const navigate = useNavigate()
  const state = location.state || {}
  const text = state.text || ''
  const results = state.results || null

  const [feedbackStatus, setFeedbackStatus] = useState('')
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false)
  const [feedbackLabel, setFeedbackLabel] = useState('')
  const [feedbackText, setFeedbackText] = useState('')

  if (!results || !text) {
    // If someone opens /results directly, send them back home
    return <Navigate to="/" replace />
  }

  const sentimentClass =
    results?.resolved?.label?.toLowerCase?.() === 'negative'
      ? 'negative'
      : results?.resolved?.label?.toLowerCase?.() === 'positive'
        ? 'positive'
        : results?.resolved?.label?.toLowerCase?.() === 'neutral'
          ? 'neutral'
          : ''

  async function submitFeedback(e) {
    e.preventDefault()
    if (!results) return
    if (!feedbackLabel) {
      setFeedbackStatus('Please select an option (correct/incorrect).')
      return
    }

    setFeedbackSubmitting(true)
    setFeedbackStatus('')

    try {
      const payload = {
        input_text: text.trim(),
        resolved_sentiment: results?.resolved?.label || '',
        confidence: results?.resolved?.confidence ?? '',
        feedback_label: feedbackLabel,
        feedback_text: feedbackText,
      }

      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      const data = await res.json()
      if (!res.ok || !data?.success) {
        setFeedbackStatus('Error submitting feedback. Please try again.')
        return
      }

      setFeedbackStatus('Thank you! Your feedback has been recorded.')
      setFeedbackLabel('')
      setFeedbackText('')
    } catch (err) {
      console.error(err)
      setFeedbackStatus('Error submitting feedback. Please try again.')
    } finally {
      setFeedbackSubmitting(false)
    }
  }

  return (
    <div className="container">
      <header>
        <h1 className="title">SentiAlign</h1>
        <p className="subtitle">Sentiment Analysis Results</p>
      </header>

      <main>
        <div className="result-section">
          <div className="analyzed-text-header">
            <h2 className="tooltip">
              Analyzed Text
              <span className="tooltip-text">
                The original input text you provided for sentiment analysis.
              </span>
            </h2>
            <button
              className="btn-back"
              onClick={() => navigate('/')}
              title="Analyze New Text"
            >
              ← Analyze New Text
            </button>
          </div>
          <div className="analyzed-text">{text.trim()}</div>
        </div>

        <div className="result-section">
          <h2 className="tooltip">
            Resolved Sentiment
            <span className="tooltip-text">
              Final sentiment predicted by the meta-classifier after combining BERT,
              RoBERTa, and Senti4SD outputs and resolving conflicts.
            </span>
          </h2>
          <div className="resolved-sentiment">
            <span className={`sentiment-label ${sentimentClass}`}>
              {results?.resolved?.label}
            </span>
            <span className="confidence">
              Confidence:{' '}
              {typeof results?.resolved?.confidence === 'number'
                ? `${(results.resolved.confidence * 100).toFixed(1)}%`
                : '-'}
            </span>
          </div>

          <div className="probabilities">
            <div className="prob-item">
              <span className="prob-label">Negative:</span>
              <span className="prob-value">
                {percent(results?.resolved?.probabilities?.negative)}
              </span>
            </div>
            <div className="prob-item">
              <span className="prob-label">Neutral:</span>
              <span className="prob-value">
                {percent(results?.resolved?.probabilities?.neutral)}
              </span>
            </div>
            <div className="prob-item">
              <span className="prob-label">Positive:</span>
              <span className="prob-value">
                {percent(results?.resolved?.probabilities?.positive)}
              </span>
            </div>
          </div>
        </div>

        <div className="result-section">
          <h2 className="tooltip">
            Base Model Predictions
            <span className="tooltip-text">
              Individual predictions from each underlying model before conflict
              resolution is applied.
            </span>
          </h2>
          <div className="base-models">
            <div className="model-result">
              <h3 className="tooltip">
                BERT
                <span className="tooltip-text">
                  Multilingual BERT model predicting sentiment using a 5-star scale
                  (1 = strong negative, 5 = strong positive).
                </span>
              </h3>
              <p className="model-label">{results?.base_predictions?.bert?.label}</p>
              <div className="confidence-bars">
                <div className="confidence-bar-item">
                  <span className="bar-label">Negative</span>
                  <div className="bar-container">
                    <div 
                      className="bar bar-negative" 
                      style={{
                        width: `${((results?.base_predictions?.bert?.probabilities?.class_1 || 0) + 
                                  (results?.base_predictions?.bert?.probabilities?.class_2 || 0)) * 100}%`
                      }}
                    />
                  </div>
                  <span className="bar-value">
                    {percent((results?.base_predictions?.bert?.probabilities?.class_1 || 0) + 
                             (results?.base_predictions?.bert?.probabilities?.class_2 || 0))}
                  </span>
                </div>
                <div className="confidence-bar-item">
                  <span className="bar-label">Neutral</span>
                  <div className="bar-container">
                    <div 
                      className="bar bar-neutral" 
                      style={{
                        width: `${(results?.base_predictions?.bert?.probabilities?.class_3 || 0) * 100}%`
                      }}
                    />
                  </div>
                  <span className="bar-value">
                    {percent(results?.base_predictions?.bert?.probabilities?.class_3)}
                  </span>
                </div>
                <div className="confidence-bar-item">
                  <span className="bar-label">Positive</span>
                  <div className="bar-container">
                    <div 
                      className="bar bar-positive" 
                      style={{
                        width: `${((results?.base_predictions?.bert?.probabilities?.class_4 || 0) + 
                                  (results?.base_predictions?.bert?.probabilities?.class_5 || 0)) * 100}%`
                      }}
                    />
                  </div>
                  <span className="bar-value">
                    {percent((results?.base_predictions?.bert?.probabilities?.class_4 || 0) + 
                             (results?.base_predictions?.bert?.probabilities?.class_5 || 0))}
                  </span>
                </div>
              </div>
            </div>

            <div className="model-result">
              <h3 className="tooltip">
                RoBERTa
                <span className="tooltip-text">
                  RoBERTa model predicting sentiment as Negative, Neutral, or Positive
                  based on Twitter-style text.
                </span>
              </h3>
              <p className="model-label">
                {results?.base_predictions?.roberta?.label}
              </p>
              <div className="confidence-bars">
                <div className="confidence-bar-item">
                  <span className="bar-label">Negative</span>
                  <div className="bar-container">
                    <div 
                      className="bar bar-negative" 
                      style={{
                        width: `${(results?.base_predictions?.roberta?.probabilities?.negative || 0) * 100}%`
                      }}
                    />
                  </div>
                  <span className="bar-value">
                    {percent(results?.base_predictions?.roberta?.probabilities?.negative)}
                  </span>
                </div>
                <div className="confidence-bar-item">
                  <span className="bar-label">Neutral</span>
                  <div className="bar-container">
                    <div 
                      className="bar bar-neutral" 
                      style={{
                        width: `${(results?.base_predictions?.roberta?.probabilities?.neutral || 0) * 100}%`
                      }}
                    />
                  </div>
                  <span className="bar-value">
                    {percent(results?.base_predictions?.roberta?.probabilities?.neutral)}
                  </span>
                </div>
                <div className="confidence-bar-item">
                  <span className="bar-label">Positive</span>
                  <div className="bar-container">
                    <div 
                      className="bar bar-positive" 
                      style={{
                        width: `${(results?.base_predictions?.roberta?.probabilities?.positive || 0) * 100}%`
                      }}
                    />
                  </div>
                  <span className="bar-value">
                    {percent(results?.base_predictions?.roberta?.probabilities?.positive)}
                  </span>
                </div>
              </div>
            </div>

            <div className="model-result">
              <h3 className="tooltip">
                Senti4SD
                <span className="tooltip-text">
                  Lexicon-based sentiment analyzer using positive and negative opinion
                  words commonly found in software engineering discussions.
                </span>
              </h3>
              <p className="model-label">
                {results?.base_predictions?.senti4sd?.label}
              </p>
              <div className="confidence-bars">
                <div className="confidence-bar-item">
                  <span className="bar-label">Negative</span>
                  <div className="bar-container">
                    <div 
                      className="bar bar-negative" 
                      style={{
                        width: `${(results?.base_predictions?.senti4sd?.probabilities?.negative || 0) * 100}%`
                      }}
                    />
                  </div>
                  <span className="bar-value">
                    {percent(results?.base_predictions?.senti4sd?.probabilities?.negative)}
                  </span>
                </div>
                <div className="confidence-bar-item">
                  <span className="bar-label">Neutral</span>
                  <div className="bar-container">
                    <div 
                      className="bar bar-neutral" 
                      style={{
                        width: `${(results?.base_predictions?.senti4sd?.probabilities?.neutral || 0) * 100}%`
                      }}
                    />
                  </div>
                  <span className="bar-value">
                    {percent(results?.base_predictions?.senti4sd?.probabilities?.neutral)}
                  </span>
                </div>
                <div className="confidence-bar-item">
                  <span className="bar-label">Positive</span>
                  <div className="bar-container">
                    <div 
                      className="bar bar-positive" 
                      style={{
                        width: `${(results?.base_predictions?.senti4sd?.probabilities?.positive || 0) * 100}%`
                      }}
                    />
                  </div>
                  <span className="bar-value">
                    {percent(results?.base_predictions?.senti4sd?.probabilities?.positive)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="result-section">
          <h2 className="tooltip">
            Explainable AI (XAI)
            <span className="tooltip-text">
              Natural language explanation of how the meta-classifier combined model
              signals to reach the final sentiment decision.
            </span>
          </h2>
          <div className="xai-section">
            <h3 className="xai-subtitle">SHAP Explanation</h3>
            <p className="xai-description">Understanding how the models arrived at their predictions</p>
            <div className="explanation">{results?.explanation || ''}</div>
          </div>
        </div>

        {Array.isArray(results?.shap_contributions) &&
        results.shap_contributions.length ? (
          <div className="result-section">
            <h2 className="tooltip">
              Feature Contributions (SHAP)
              <span className="tooltip-text">
                SHAP values show how each model probability (feature) pushed the
                prediction toward or away from the final sentiment label.
              </span>
            </h2>
            <div className="shap-contributions">
              <table className="shap-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Contribution</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {results.shap_contributions.map((c) => (
                    <tr key={`${c.feature}-${c.shap_value}`}>
                      <td>
                        <code>{c.feature}</code>
                      </td>
                      <td
                        className={`shap-value ${
                          c.shap_value > 0 ? 'positive' : 'negative'
                        }`}
                      >
                        {typeof c.shap_value === 'number'
                          ? c.shap_value.toFixed(4)
                          : String(c.shap_value)}
                      </td>
                      <td>{c.meaning}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}

        <div className="result-section">
          <h2>Your Feedback</h2>
          <p className="feedback-instruction">
            Was the resolved sentiment correct for the given text?
          </p>

          <form id="feedback-form" onSubmit={submitFeedback}>
            <div className="feedback-options">
              <label>
                <input
                  type="radio"
                  name="feedback_label"
                  value="correct"
                  checked={feedbackLabel === 'correct'}
                  onChange={() => setFeedbackLabel('correct')}
                />{' '}
                Yes, the result is correct
              </label>

              <label>
                <input
                  type="radio"
                  name="feedback_label"
                  value="incorrect"
                  checked={feedbackLabel === 'incorrect'}
                  onChange={() => setFeedbackLabel('incorrect')}
                />{' '}
                No, the result is incorrect
              </label>
            </div>

            <div className="feedback-comment">
              <label htmlFor="feedback_text">Optional comment:</label>
              <textarea
                id="feedback_text"
                name="feedback_text"
                placeholder="Explain why you think the result is correct or incorrect..."
                value={feedbackText}
                onChange={(e) => setFeedbackText(e.target.value)}
              />
            </div>

            <div className="row">
              <button className="btn-primary" type="submit" disabled={feedbackSubmitting}>
                {feedbackSubmitting ? 'Submitting…' : 'Submit Feedback'}
              </button>
              <button
                className="btn-primary"
                type="button"
                onClick={() => navigate('/')}
              >
                Analyze Another Text
              </button>
      </div>

            {feedbackStatus ? (
              <p
                className="feedback-status"
                style={{
                  color: feedbackStatus.startsWith('Thank') ? 'green' : 'red',
                }}
              >
                {feedbackStatus}
              </p>
            ) : null}
          </form>
        </div>
      </main>
    </div>
  )
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<AnalyzePage />} />
      <Route path="/results" element={<ResultsPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
