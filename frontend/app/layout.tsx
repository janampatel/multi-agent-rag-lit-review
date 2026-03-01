import './globals.css'

export const metadata = {
  title: 'Multi-Agent RAG System',
  description: 'Systematic Literature Review with AI Agents',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 min-h-screen">
        <div className="container mx-auto px-4 py-8">
          <header className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900">Multi-Agent RAG System</h1>
            <p className="text-gray-600">Systematic Literature Review with AI Agents</p>
          </header>
          {children}
        </div>
      </body>
    </html>
  )
}