import collections
import json

import elements
import numpy as np


def final_eval(driver, policy, init_policy, logger, logdir, episodes):
  """Fixed-episode deterministic evaluation -> the reported benchmark number.

  Runs `episodes` episodes through `driver` (whose envs must be on the held-out
  / val partition) with the given (eval-mode) `policy`, then logs the result two
  ways:

    - `eval/score`, `eval/length`, `eval/episodes`: the final point on the same
      curve as periodic eval. `episodes` jumps from the periodic count to this
      one, which is the only field marking it as the final eval -- this matches
      how HRSSM / VIBES log theirs, for a directly comparable combined plot.

    - `final_eval/{score_mean,score_std,score_ci95,score_min,score_max,
      length_mean,episodes}` plus `<logdir>/final_eval.json` (the summary and
      every raw episode score): the unambiguous number with its 95% CI, which
      the single curve point cannot carry.

  Appends its own collector to `driver` and resets it; the caller owns driver
  construction and teardown. Returns the summary dict.
  """
  n = int(episodes)
  scores, lengths = [], []
  aggs = collections.defaultdict(elements.Agg)

  def collect(tran, worker):
    ep = aggs[worker]
    tran['is_first'] and ep.reset()
    ep.add('score', tran['reward'], agg='sum')
    ep.add('length', 1, agg='sum')
    if tran['is_last']:
      res = ep.result()
      scores.append(float(res['score']))
      lengths.append(float(res['length']))
      print(f'  episode {len(scores)}/{n}: score {scores[-1]:.1f}')

  driver.on_step(collect)
  driver.reset(init_policy)
  driver(policy, episodes=n)

  # Driver stops on the first step where the episode count reaches n, so a few
  # workers can finish together and overshoot; keep exactly the first n.
  s = np.array(scores[:n])
  lens = np.array(lengths[:n])
  ci95 = float(1.96 * s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0
  summary = {
      'score_mean': float(s.mean()),
      'score_std': float(s.std(ddof=1)) if len(s) > 1 else 0.0,
      'score_ci95': ci95,
      'score_min': float(s.min()),
      'score_max': float(s.max()),
      'length_mean': float(lens.mean()),
      'episodes': int(len(s)),
  }
  logger.add({
      'score': summary['score_mean'],
      'length': summary['length_mean'],
      'episodes': summary['episodes'],
  }, prefix='eval')
  logger.add(summary, prefix='final_eval')
  logger.write()
  print('Final eval summary:', summary)
  try:
    (logdir / 'final_eval.json').write(
        json.dumps({**summary, 'scores': s.tolist()}, indent=2))
  except Exception as e:
    print('Could not write final_eval.json:', e)
  return summary
