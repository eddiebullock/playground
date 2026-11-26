'use client';

import { useState, useMemo } from 'react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

export type Column<T> = {
  key: keyof T;
  header: string | React.ReactNode;
  render?: (value: T[keyof T], item: T) => React.ReactNode;
  sortable?: boolean;
  className?: string;
};

export type FilterOption = {
  label: string;
  value: string;
};

interface CollapsibleTableProps<T> {
  data: T[];
  columns: Column<T>[];
  title: string;
  filterOptions?: FilterOption[];
  defaultFilter?: string;
  onFilterChange?: (filter: string) => void;
  className?: string;
}

export default function CollapsibleTable<T extends { id: string | number }>({
  data,
  columns,
  title,
  filterOptions,
  defaultFilter,
  onFilterChange,
  className = '',
}: CollapsibleTableProps<T>) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [sortConfig, setSortConfig] = useState<{
    key: keyof T | null;
    direction: 'asc' | 'desc';
  }>({ key: null, direction: 'asc' });
  const [activeFilter, setActiveFilter] = useState(defaultFilter || '');

  const sortedData = useMemo(() => {
    if (!sortConfig.key) return data;

    return [...data].sort((a, b) => {
      const aValue = a[sortConfig.key!];
      const bValue = b[sortConfig.key!];

      if (aValue === bValue) return 0;
      if (aValue === null || aValue === undefined) return 1;
      if (bValue === null || bValue === undefined) return -1;

      const comparison = aValue < bValue ? -1 : 1;
      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });
  }, [data, sortConfig]);

  const handleSort = (key: keyof T) => {
    setSortConfig((current) => ({
      key,
      direction:
        current.key === key && current.direction === 'asc' ? 'desc' : 'asc',
    }));
  };

  const handleFilterChange = (value: string) => {
    setActiveFilter(value);
    onFilterChange?.(value);
  };

  return (
    <div className={`bg-white rounded-xl shadow-lg border border-gray-100 ${className}`}>
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-gray-900">{title}</h3>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
            aria-label={isExpanded ? 'Collapse table' : 'Expand table'}
          >
            {isExpanded ? (
              <ChevronUpIcon className="h-5 w-5 text-gray-500" />
            ) : (
              <ChevronDownIcon className="h-5 w-5 text-gray-500" />
            )}
          </button>
        </div>
        
        {filterOptions && (
          <div className="flex flex-wrap gap-2">
            {filterOptions.map((option) => (
              <button
                key={option.value}
                onClick={() => handleFilterChange(option.value)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeFilter === option.value
                    ? 'bg-indigo-100 text-indigo-800 border border-indigo-200'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-200'
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {isExpanded && (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                {columns.map((column) => (
                  <th
                    key={String(column.key)}
                    scope="col"
                    className={`px-6 py-4 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider ${
                      column.className || ''
                    }`}
                  >
                    {column.sortable && typeof column.header === 'string' ? (
                      <button
                        onClick={() => handleSort(column.key)}
                        className="group inline-flex items-center hover:text-indigo-600 transition-colors"
                      >
                        {column.header}
                        <span className="ml-2 flex-none rounded">
                          {sortConfig.key === column.key && (
                            <span className="text-indigo-600 font-bold">
                              {sortConfig.direction === 'asc' ? '↑' : '↓'}
                            </span>
                          )}
                        </span>
                      </button>
                    ) : (
                      column.header
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {sortedData.length === 0 ? (
                <tr>
                  <td colSpan={columns.length} className="px-6 py-8 text-center text-gray-500">
                    No articles found matching the current criteria.
                  </td>
                </tr>
              ) : (
                sortedData.map((item, index) => (
                  <tr key={item.id} className={`hover:bg-gray-50 transition-colors ${index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}`}>
                    {columns.map((column) => (
                      <td
                        key={String(column.key)}
                        className={`px-6 py-4 text-sm text-gray-900 ${
                          column.className || ''
                        }`}
                      >
                        {column.render
                          ? column.render(item[column.key], item)
                          : String(item[column.key])}
                      </td>
                    ))}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
} 