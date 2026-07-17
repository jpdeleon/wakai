/* Numerical utilities shared by the age-method views and joint summary. */
(function attachPosteriorUtilities(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WakaiPosterior = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function posteriorFactory() {
  function finiteNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : 0;
  }

  function integrate(grid, values) {
    if (!Array.isArray(grid) || !Array.isArray(values) || grid.length !== values.length || grid.length < 2) {
      return 0;
    }
    let total = 0;
    for (let i = 1; i < grid.length; i += 1) {
      const width = finiteNumber(grid[i]) - finiteNumber(grid[i - 1]);
      if (width <= 0) continue;
      total += 0.5 * width * (Math.max(0, finiteNumber(values[i - 1])) + Math.max(0, finiteNumber(values[i])));
    }
    return total;
  }

  function normalize(grid, values) {
    const clean = values.map(value => Math.max(0, finiteNumber(value)));
    const area = integrate(grid, clean);
    return area > 0 ? clean.map(value => value / area) : clean.map(() => 0);
  }

  function interpolatePDF(sourceGrid, sourcePDF, targetGrid) {
    if (!Array.isArray(sourceGrid) || !Array.isArray(sourcePDF) || sourceGrid.length !== sourcePDF.length || sourceGrid.length < 2) {
      throw new Error('Posterior grid and PDF must contain matching samples');
    }

    const pairs = sourceGrid
      .map((age, index) => [Number(age), Math.max(0, finiteNumber(sourcePDF[index]))])
      .filter(([age]) => Number.isFinite(age))
      .sort((left, right) => left[0] - right[0]);
    if (pairs.length < 2) throw new Error('Posterior grid needs at least two finite ages');

    const firstAge = pairs[0][0];
    const lastAge = pairs[pairs.length - 1][0];
    let cursor = 0;
    const resampled = targetGrid.map(rawAge => {
      const age = Number(rawAge);
      // A PDF has no implied support outside the grid returned by its model.
      // Zero-padding avoids manufacturing an old-age tail from the last bin.
      if (!Number.isFinite(age) || age < firstAge || age > lastAge) return 0;
      while (cursor < pairs.length - 2 && pairs[cursor + 1][0] < age) cursor += 1;
      const [x0, y0] = pairs[cursor];
      const [x1, y1] = pairs[cursor + 1];
      if (x1 === x0) return y0;
      return y0 + ((age - x0) / (x1 - x0)) * (y1 - y0);
    });
    return normalize(targetGrid, resampled);
  }

  function percentile(grid, cumulative, probability) {
    for (let i = 1; i < cumulative.length; i += 1) {
      if (cumulative[i] < probability) continue;
      const span = cumulative[i] - cumulative[i - 1];
      if (span <= 0) return grid[i];
      const fraction = (probability - cumulative[i - 1]) / span;
      return grid[i - 1] + fraction * (grid[i] - grid[i - 1]);
    }
    return grid[grid.length - 1];
  }

  function stats(grid, values) {
    const pdf = normalize(grid, values);
    const area = integrate(grid, pdf);
    if (area <= 0) return null;

    const cumulative = new Array(grid.length).fill(0);
    for (let i = 1; i < grid.length; i += 1) {
      const width = grid[i] - grid[i - 1];
      cumulative[i] = cumulative[i - 1] + 0.5 * width * (pdf[i - 1] + pdf[i]);
    }
    const total = cumulative[cumulative.length - 1];
    const normalizedCDF = cumulative.map(value => value / total);
    return {
      lower: percentile(grid, normalizedCDF, 0.16),
      median: percentile(grid, normalizedCDF, 0.50),
      upper: percentile(grid, normalizedCDF, 0.84),
    };
  }

  function combineIndependent(grid, leftPDF, rightPDF) {
    if (leftPDF.length !== grid.length || rightPDF.length !== grid.length) {
      throw new Error('Joint posterior inputs must share the same age grid');
    }
    const product = grid.map((_, index) => Math.max(0, finiteNumber(leftPDF[index])) * Math.max(0, finiteNumber(rightPDF[index])));
    const peak = Math.max(...product);
    if (!(peak > 0)) {
      return { pdf: null, stats: null, status: 'no-overlap' };
    }
    // Scale before integrating to avoid underflow when both densities are tiny.
    const pdf = normalize(grid, product.map(value => value / peak));
    return { pdf, stats: stats(grid, pdf), status: 'ok' };
  }

  return { integrate, normalize, interpolatePDF, stats, combineIndependent };
});
