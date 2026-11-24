import { randomUUID } from 'crypto';
import { promises as fs } from 'fs';
import path from 'path';
import { NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';

interface Manifest {
  round_id: string;
  step: number;
  samples: { id: string; prompt: string; group: number; seed: string }[];
}

async function readActiveRound(jobID: string) {
  const job = await prisma.job.findUnique({ where: { id: jobID }, select: { name: true, job_config: true } });
  if (!job || JSON.parse(job.job_config).config?.process?.[0]?.type !== 'moss_image_sifr') return null;
  const trainingRoot = await getTrainingFolder();
  const jobFolder = path.resolve(trainingRoot, job.name);
  if (path.dirname(jobFolder) !== path.resolve(trainingRoot)) return null;
  const root = path.join(jobFolder, 'sifr');
  let manifest: Manifest;
  try {
    manifest = JSON.parse(await fs.readFile(path.join(root, 'active.json'), 'utf8'));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null;
    throw error;
  }
  if (
    !/^[a-f0-9]{32}$/.test(manifest.round_id) ||
    !Number.isSafeInteger(manifest.step) ||
    manifest.step < 0 ||
    !manifest.samples.every(sample => /^sample-\d{4,}$/.test(sample.id))
  ) {
    throw new Error('Invalid SIFR round manifest');
  }
  const directory = path.join(root, `round-${String(manifest.step).padStart(9, '0')}`);
  return { manifest, directory, scoresPath: path.join(directory, 'scores.json') };
}

export async function GET(_request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const round = await readActiveRound(jobID);
  if (!round) return NextResponse.json({ round: null });
  const submitted = await fs.stat(round.scoresPath).then(
    () => true,
    error => {
      if (error.code === 'ENOENT') return false;
      throw error;
    },
  );
  return NextResponse.json({
    round: {
      id: round.manifest.round_id,
      step: round.manifest.step,
      submitted,
      samples: round.manifest.samples.map(sample => ({
        ...sample,
        imagePath: path.join(round.directory, `${sample.id}.png`),
      })),
    },
  });
}

export async function POST(request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const round = await readActiveRound(jobID);
  const body = await request.json().catch(() => null);
  if (!round || body?.round_id !== round.manifest.round_id) {
    return NextResponse.json({ error: 'This round is no longer active.' }, { status: 409 });
  }
  const ids = round.manifest.samples.map(sample => sample.id);
  const scores = body.scores;
  if (
    !scores ||
    typeof scores !== 'object' ||
    Array.isArray(scores) ||
    Object.keys(scores).length !== ids.length ||
    !ids.every(id => Object.hasOwn(scores, id) && Number.isInteger(scores[id]) && scores[id] >= 1 && scores[id] <= 5)
  ) {
    return NextResponse.json({ error: 'Rate every image from 1 to 5.' }, { status: 400 });
  }

  // Publish a complete file exactly once, including when two tabs submit concurrently.
  const temporary = path.join(round.directory, `.scores-${randomUUID()}.tmp`);
  try {
    await fs.writeFile(temporary, JSON.stringify({ round_id: body.round_id, scores }), { flag: 'wx' });
    await fs.link(temporary, round.scoresPath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'EEXIST') {
      return NextResponse.json({ error: 'Ratings have already been submitted for this round.' }, { status: 409 });
    }
    throw error;
  } finally {
    await fs.rm(temporary, { force: true });
  }
  return NextResponse.json({ submitted: true });
}
