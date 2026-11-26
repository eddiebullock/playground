export default function BrainLoadingState() {
    return (
        <div className="absolute inset-0 bg-white/80 backdrop-blur-sm flex flex-col items-center justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4" />
            <p className="text-gray-600">Loading brain visualization...</p>
            <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
        </div>
    );
} 