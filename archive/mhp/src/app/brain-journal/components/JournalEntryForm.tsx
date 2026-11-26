'use client';

import { useState } from 'react';

interface JournalEntryFormProps {
    onSubmit: (entry: string) => Promise<void>;
    isSubmitting: boolean;
}

export default function JournalEntryForm({ onSubmit, isSubmitting }: JournalEntryFormProps) {
    const [entry, setEntry] = useState('');

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault();
        
        // Validate entry
        if (!entry.trim()) {
            console.error('Journal entry is empty');
            return;
        }

        try {
            await onSubmit(entry.trim());
            // Clear form after successful submission
            setEntry('');
        } catch (error) {
            console.error('Error submitting journal entry:', error);
        }
    }

    return (
        <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Write Your Journal Entry
            </h2>
            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label 
                        htmlFor="journal-entry" 
                        className="block text-sm font-medium text-gray-700 mb-2"
                    >
                        How are you feeling today?
                    </label>
                    <textarea
                        id="journal-entry"
                        rows={8}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        placeholder="Describe your thoughts, feelings, and experiences..."
                        value={entry}
                        onChange={(e) => setEntry(e.target.value)}
                        disabled={isSubmitting}
                        required
                    />
                </div>
                <div className="flex justify-end">
                    <button
                        type="submit"
                        disabled={isSubmitting || !entry.trim()}
                        className={`
                            px-4 py-2 rounded-md text-white font-medium
                            ${isSubmitting || !entry.trim()
                                ? 'bg-gray-400 cursor-not-allowed'
                                : 'bg-blue-600 hover:bg-blue-700'
                            }
                        `}
                    >
                        {isSubmitting ? 'Analyzing...' : 'Analyze Entry'}
                    </button>
                </div>
            </form>
        </div>
    );
} 