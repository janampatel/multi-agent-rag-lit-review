'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:8000'

interface Document {
  id: string
  filename: string
  pages: number
  chunks: number
  status: string
}

interface JobStatus {
  job_id: string
  status: string
  progress: number
  current_step: string
  result: string
  error: string
}

function formatMarkdown(text: string): string {
  // Create a temporary element to decode HTML entities
  const textarea = document.createElement('textarea');
  textarea.innerHTML = text;
  const decoded = textarea.value;
  
  return decoded
    .replace(/^# (.+)$/gm, '<h1 class="text-3xl font-bold mb-4 mt-6 text-gray-900">$1</h1>')
    .replace(/^## (.+)$/gm, '<h2 class="text-2xl font-semibold mt-8 mb-4 text-gray-800 border-b-2 border-blue-500 pb-2">$1</h2>')
    .replace(/^### (.+)$/gm, '<h3 class="text-xl font-medium mt-6 mb-3 text-gray-700">$1</h3>')
    .replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold text-gray-900">$1</strong>')
    .replace(/\*(.+?)\*/g, '<em class="italic">$1</em>')
    .replace(/\[([0-9]+)\]/g, '<sup class="text-blue-600 font-bold text-sm">[<span class="hover:underline cursor-pointer">$1</span>]</sup>')
    .split('\n\n')
    .map(para => para.trim() ? `<p class="mb-4 text-gray-700 leading-relaxed text-base">${para}</p>` : '')
    .join('');
}

export default function Home() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [uploading, setUploading] = useState(false)
  const [query, setQuery] = useState('Methods for unlearning in federated learning')
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null)
  const [polling, setPolling] = useState(false)

  useEffect(() => {
    fetchDocuments()
  }, [])

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (polling && currentJob?.job_id) {
      interval = setInterval(() => {
        checkJobStatus(currentJob.job_id)
      }, 2000)
    }
    return () => clearInterval(interval)
  }, [polling, currentJob?.job_id])

  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/documents`)
      setDocuments(response.data.documents)
    } catch (error) {
      console.error('Failed to fetch documents:', error)
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      await axios.post(`${API_BASE}/api/documents/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      fetchDocuments()
    } catch (error) {
      console.error('Upload failed:', error)
    } finally {
      setUploading(false)
    }
  }

  const startResearch = async () => {
    try {
      const response = await axios.post(`${API_BASE}/api/research/start`, {
        query,
        use_arxiv: false,
        threshold: 3
      })
      
      setCurrentJob({
        job_id: response.data.job_id,
        status: 'running',
        progress: 0,
        current_step: 'Starting...',
        result: '',
        error: ''
      })
      setPolling(true)
    } catch (error) {
      console.error('Failed to start research:', error)
    }
  }

  const checkJobStatus = async (jobId: string) => {
    try {
      const response = await axios.get(`${API_BASE}/api/research/status/${jobId}`)
      setCurrentJob(response.data)
      
      if (response.data.status === 'completed' || response.data.status === 'failed') {
        setPolling(false)
      }
    } catch (error) {
      console.error('Failed to check job status:', error)
      setPolling(false)
    }
  }

  return (
    <div className="space-y-8">
      {/* Document Upload Section */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Document Management</h2>
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Upload PDF Documents
          </label>
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            disabled={uploading}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
          {uploading && (
            <div className="mt-2 text-blue-600">
              <div className="loading-spinner inline-block w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full mr-2"></div>
              Processing document...
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {documents.map((doc) => (
            <div key={doc.id} className="border rounded-lg p-4">
              <h3 className="font-medium text-gray-900 truncate">{doc.filename}</h3>
              <p className="text-sm text-gray-600">{doc.pages} pages, {doc.chunks} chunks</p>
              <span className="inline-block mt-2 px-2 py-1 text-xs bg-green-100 text-green-800 rounded">
                {doc.status}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Research Section */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Research Query</h2>
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Research Question
          </label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            rows={3}
            placeholder="Enter your research question..."
          />
        </div>

        <button
          onClick={startResearch}
          disabled={!query.trim() || documents.length === 0 || polling}
          className="bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium transition-all flex items-center gap-2"
        >
          {polling && (
            <div className="loading-spinner inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
          )}
          {polling ? 'Research in Progress...' : 'Start Research'}
        </button>
      </div>

      {/* Progress Section */}
      {currentJob && (
        <div className="bg-white p-6 rounded-lg shadow animate-fadeIn">
          <h2 className="text-xl font-semibold mb-4">Research Progress</h2>
          
          {currentJob.status === 'running' && (
            <div className="mb-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded animate-pulse">
              <div className="flex items-center gap-3">
                <div className="loading-spinner w-6 h-6 border-3 border-blue-600 border-t-transparent rounded-full"></div>
                <div>
                  <p className="font-medium text-blue-900">AI Agents Working...</p>
                  <p className="text-sm text-blue-700">Analyzing documents and synthesizing insights</p>
                </div>
              </div>
            </div>
          )}
          
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span className="flex items-center font-medium">
                {currentJob.status === 'running' && (
                  <div className="loading-spinner inline-block w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full mr-2"></div>
                )}
                {currentJob.current_step}
              </span>
              <span className="font-bold text-blue-600">{Math.round(currentJob.progress * 100)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 shadow-inner">
              <div 
                className="bg-gradient-to-r from-blue-500 via-blue-600 to-indigo-600 h-3 rounded-full progress-bar shadow-sm relative overflow-hidden"
                style={{ width: `${currentJob.progress * 100}%` }}
              >
                <div className="absolute inset-0 bg-white opacity-20 animate-shimmer"></div>
              </div>
            </div>
          </div>

          {currentJob.status === 'completed' && currentJob.result && (
            <div className="mt-6 animate-fadeIn">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Research Results</h3>
                <span className="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full">✓ Completed</span>
              </div>
              <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-6 rounded-lg border border-gray-200 max-h-[600px] overflow-y-auto">
                <div 
                  className="prose prose-sm max-w-none"
                  dangerouslySetInnerHTML={{ __html: formatMarkdown(currentJob.result) }}
                />
              </div>
            </div>
          )}

          {currentJob.status === 'failed' && currentJob.error && (
            <div className="mt-4 p-4 bg-red-50 border-l-4 border-red-500 rounded-md animate-shake">
              <p className="text-red-800 font-medium">⚠ Error: {currentJob.error}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}