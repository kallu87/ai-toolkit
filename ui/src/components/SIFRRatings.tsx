'use client';

import { useState } from 'react';
import { Job } from '@prisma/client';
import usePollLoop from '@/hooks/usePollLoop';
import { apiClient } from '@/utils/api';
import { encodeFilePathForUrl } from '@/utils/basic';

interface Round {
  id: string;
  step: number;
  submitted: boolean;
  samples: { id: string; prompt: string; group: number; imagePath: string }[];
}

function RatingForm({ round, endpoint }: { round: Round; endpoint: string }) {
  const [scores, setScores] = useState<Record<string, number>>({});
  const [status, setStatus] = useState<'editing' | 'submitting' | 'submitted'>('editing');
  const [error, setError] = useState('');
  const rated = Object.keys(scores).length;
  const submitted = round.submitted || status === 'submitted';

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (rated !== round.samples.length || submitted || status === 'submitting') return;
    setStatus('submitting');
    setError('');
    try {
      await apiClient.post(endpoint, { round_id: round.id, scores });
      setStatus('submitted');
    } catch (error: any) {
      setError(error.response?.data?.error || 'Could not submit ratings. Try again.');
      setStatus('editing');
    }
  }

  if (submitted)
    return <p role="status">Ratings submitted for step {round.step}. Training will resume automatically.</p>;

  const groups = new Map<number, Round['samples']>();
  for (const sample of round.samples) {
    const group = groups.get(sample.group) ?? [];
    group.push(sample);
    groups.set(sample.group, group);
  }

  return (
    <form onSubmit={submit} className="mx-auto max-w-6xl space-y-6 pb-8">
      <div>
        <h2 className="text-xl font-semibold">Rate samples from step {round.step}</h2>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
          Rate each image from 1 (worst) to 5 (best) for its prompt.
        </p>
      </div>
      {Array.from(groups, ([group, samples]) => (
        <fieldset key={group} className="space-y-3" disabled={status === 'submitting'}>
          <legend className="mb-2 font-medium">{samples[0].prompt}</legend>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {samples.map((sample, index) => {
              const url = `/api/img/${encodeFilePathForUrl(sample.imagePath)}`;
              return (
                <div key={sample.id} className="overflow-hidden rounded-lg border border-gray-300 dark:border-gray-700">
                  <a href={url} target="_blank" rel="noreferrer">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={url}
                      alt={`Sample ${index + 1} for ${sample.prompt}`}
                      className="w-full object-contain"
                      loading="lazy"
                    />
                  </a>
                  <label className="flex items-center justify-between gap-2 p-3 text-sm">
                    <span>Sample {index + 1} rating</span>
                    <select
                      required
                      className="rounded border border-gray-400 bg-white px-2 py-1 text-gray-900 dark:border-gray-600 dark:bg-gray-800 dark:text-white"
                      value={scores[sample.id] ?? ''}
                      onChange={event =>
                        setScores(previous => ({ ...previous, [sample.id]: Number(event.target.value) }))
                      }
                    >
                      <option value="" disabled>
                        Rate
                      </option>
                      {[1, 2, 3, 4, 5].map(score => (
                        <option key={score} value={score}>
                          {score}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              );
            })}
          </div>
        </fieldset>
      ))}
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="submit"
          disabled={rated !== round.samples.length || status === 'submitting'}
          className="rounded bg-blue-600 px-4 py-2 font-medium text-white disabled:opacity-50"
        >
          {status === 'submitting' ? 'Submitting…' : 'Submit ratings'}
        </button>
        <span aria-live="polite">
          {rated} of {round.samples.length} rated
        </span>
        {error && <p role="alert">{error}</p>}
      </div>
    </form>
  );
}

export default function SIFRRatings({ job }: { job: Job }) {
  const [round, setRound] = useState<Round | null>();
  const [loadError, setLoadError] = useState(false);
  const endpoint = `/api/jobs/${job.id}/ratings`;
  usePollLoop(
    async () => {
      try {
        const response = await apiClient.get<{ round: Round | null }>(endpoint);
        setRound(response.data.round);
        setLoadError(false);
      } catch {
        setLoadError(true);
      }
    },
    3000,
    [endpoint],
  );
  if (loadError && !round) return <p role="alert">Could not load this job’s ratings.</p>;
  if (round === undefined) return <p>Loading ratings…</p>;
  if (!round) return <p>No samples are waiting for ratings. {job.info}</p>;
  return <RatingForm key={round.id} round={round} endpoint={endpoint} />;
}
