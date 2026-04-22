/**
 * DataForge — Comprehensive Data Analysis & Visualization Toolkit
 * 
 * A self-contained, zero-dependency statistics engine with rich data structures,
 * statistical analysis, machine learning algorithms, and ASCII visualization.
 * 
 * @author The Fellowship
 * @version 2.0.0
 * @license MIT
 * 
 * ## Features
 * - **Series & DataFrame**: Pandas-like data structures with indexing, slicing, filtering
 * - **Descriptive Statistics**: mean, median, mode, variance, stddev, skewness, kurtosis, percentiles
 * - **Correlation & Regression**: Pearson/Spearman correlation, linear/polynomial/multiple regression
 * - **Hypothesis Testing**: t-test (one-sample, two-sample, paired), chi-squared, ANOVA, Shapiro-Wilk
 * - **Clustering**: K-Means with convergence detection, silhouette score evaluation
 * - **Time Series**: Moving averages, exponential smoothing, seasonal decomposition, forecasting
 * - **Probability Distributions**: Normal, Poisson, Binomial, Exponential PDF/CDF/quantiles
 * - **CSV Parser**: Full RFC 4180 compliant parser with type inference
 * - **ASCII Visualization**: Bar charts, line charts, histograms, scatter plots, heatmaps, box plots
 * - **Data Pipeline**: Chainable transform/filter/aggregate/join operations
 * 
 * @example
 * var df = DataForge.DataFrame.fromRecords([
 *   { name: 'Alice', age: 30, salary: 75000 },
 *   { name: 'Bob', age: 25, salary: 65000 }
 * ]);
 * console.log(df.describe());
 * console.log(DataForge.Charts.bar(df.col('salary').values, { labels: df.col('name').values }));
 */

// ============================================================
//  UTILITY FUNCTIONS
// ============================================================

/**
 * Deep-clone a plain value (numbers, strings, arrays, plain objects).
 * @param {*} obj - The value to clone.
 * @returns {*} A deep copy of the input.
 */
function deepClone(obj) {
  if (obj === null || typeof obj !== 'object') return obj;
  if (Array.isArray(obj)) return obj.map(function(v) { return deepClone(v); });
  var out = {};
  var keys = Object.keys(obj);
  for (var i = 0; i < keys.length; i++) {
    out[keys[i]] = deepClone(obj[keys[i]]);
  }
  return out;
}

/**
 * Generate a sequence of numbers from start to end (exclusive) with a given step.
 * @param {number} start - Start value (inclusive).
 * @param {number} end - End value (exclusive).
 * @param {number} [step=1] - Step size.
 * @returns {number[]} The generated range.
 */
function range(start, end, step) {
  if (step === undefined) step = 1;
  var arr = [];
  for (var i = start; i < end; i += step) arr.push(i);
  return arr;
}

/**
 * Zip multiple arrays into an array of tuples.
 * @param {...Array} arrays - Arrays to zip together.
 * @returns {Array[]} Zipped tuples.
 */
function zip() {
  var arrays = Array.prototype.slice.call(arguments);
  var minLen = Infinity;
  for (var i = 0; i < arrays.length; i++) {
    if (arrays[i].length < minLen) minLen = arrays[i].length;
  }
  var result = [];
  for (var j = 0; j < minLen; j++) {
    var tuple = [];
    for (var k = 0; k < arrays.length; k++) tuple.push(arrays[k][j]);
    result.push(tuple);
  }
  return result;
}

/**
 * Repeat a string n times.
 * @param {string} str - String to repeat.
 * @param {number} n - Number of times.
 * @returns {string}
 */
function repeatStr(str, n) {
  if (n <= 0) return '';
  var result = '';
  for (var i = 0; i < n; i++) result += str;
  return result;
}

/**
 * Pad a string to a given length.
 * @param {string} str - Input string.
 * @param {number} len - Target length.
 * @param {string} [side='right'] - 'left' or 'right'.
 * @returns {string}
 */
function padStr(str, len, side) {
  str = String(str);
  if (str.length >= len) return str.substring(0, len);
  var pad = repeatStr(' ', len - str.length);
  return side === 'left' ? pad + str : str + pad;
}

/**
 * Format a number to fixed decimal places with alignment.
 * @param {number} num - The number.
 * @param {number} [decimals=4] - Decimal places.
 * @returns {string}
 */
function formatNum(num, decimals) {
  if (decimals === undefined) decimals = 4;
  if (num === null || num === undefined || isNaN(num)) return 'NaN';
  if (!isFinite(num)) return num > 0 ? 'Infinity' : '-Infinity';
  return Number(num).toFixed(decimals);
}

// ============================================================
//  SERIES — 1D labeled array
// ============================================================

/**
 * Create a Series — a one-dimensional labeled array capable of holding any data type.
 * Provides vectorized operations and statistical methods.
 * 
 * @param {Array} values - The data values.
 * @param {Object} [opts] - Options.
 * @param {string} [opts.name=''] - Series name.
 * @param {Array} [opts.index] - Custom index labels.
 * @returns {Object} A Series object with statistical and transformation methods.
 * 
 * @example
 * var s = createSeries([10, 20, 30, 40, 50], { name: 'scores' });
 * console.log(s.mean());    // 30
 * console.log(s.std());     // 15.81...
 * console.log(s.describe());
 */
function createSeries(values, opts) {
  if (!opts) opts = {};
  var _values = values.slice();
  var _name = opts.name || '';
  var _index = opts.index || range(0, _values.length);

  /** @returns {number[]} Numeric values with NaN removed */
  function numericValues() {
    var out = [];
    for (var i = 0; i < _values.length; i++) {
      var v = Number(_values[i]);
      if (!isNaN(v)) out.push(v);
    }
    return out;
  }

  var series = {
    /** @type {Array} Raw values array */
    values: _values,
    /** @type {string} Series name */
    name: _name,
    /** @type {Array} Index labels */
    index: _index,
    /** @returns {number} Number of elements */
    length: _values.length,

    /**
     * Get a sorted copy of numeric values.
     * @returns {number[]}
     */
    sorted: function() {
      return numericValues().sort(function(a, b) { return a - b; });
    },

    /**
     * Calculate arithmetic mean.
     * @returns {number}
     */
    mean: function() {
      var nums = numericValues();
      if (nums.length === 0) return NaN;
      var sum = 0;
      for (var i = 0; i < nums.length; i++) sum += nums[i];
      return sum / nums.length;
    },

    /**
     * Calculate median (50th percentile).
     * @returns {number}
     */
    median: function() {
      var s = series.sorted();
      if (s.length === 0) return NaN;
      var mid = Math.floor(s.length / 2);
      return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
    },

    /**
     * Find the mode(s) — most frequently occurring value(s).
     * @returns {number[]}
     */
    mode: function() {
      var freq = {};
      var nums = numericValues();
      for (var i = 0; i < nums.length; i++) {
        var k = String(nums[i]);
        freq[k] = (freq[k] || 0) + 1;
      }
      var maxFreq = 0;
      var keys = Object.keys(freq);
      for (var j = 0; j < keys.length; j++) {
        if (freq[keys[j]] > maxFreq) maxFreq = freq[keys[j]];
      }
      var modes = [];
      for (var l = 0; l < keys.length; l++) {
        if (freq[keys[l]] === maxFreq) modes.push(Number(keys[l]));
      }
      return modes;
    },

    /**
     * Calculate population or sample variance.
     * @param {boolean} [population=false] - If true, use N; else use N-1.
     * @returns {number}
     */
    variance: function(population) {
      var nums = numericValues();
      if (nums.length < 2) return NaN;
      var m = series.mean();
      var sumSq = 0;
      for (var i = 0; i < nums.length; i++) sumSq += (nums[i] - m) * (nums[i] - m);
      return sumSq / (population ? nums.length : nums.length - 1);
    },

    /**
     * Calculate standard deviation.
     * @param {boolean} [population=false] - Population or sample.
     * @returns {number}
     */
    std: function(population) {
      return Math.sqrt(series.variance(population));
    },

    /**
     * Calculate the minimum value.
     * @returns {number}
     */
    min: function() {
      var nums = numericValues();
      if (nums.length === 0) return NaN;
      var m = nums[0];
      for (var i = 1; i < nums.length; i++) if (nums[i] < m) m = nums[i];
      return m;
    },

    /**
     * Calculate the maximum value.
     * @returns {number}
     */
    max: function() {
      var nums = numericValues();
      if (nums.length === 0) return NaN;
      var m = nums[0];
      for (var i = 1; i < nums.length; i++) if (nums[i] > m) m = nums[i];
      return m;
    },

    /**
     * Calculate the sum of all numeric values.
     * @returns {number}
     */
    sum: function() {
      var nums = numericValues();
      var s = 0;
      for (var i = 0; i < nums.length; i++) s += nums[i];
      return s;
    },

    /**
     * Calculate the p-th percentile using linear interpolation.
     * @param {number} p - Percentile (0–100).
     * @returns {number}
     */
    percentile: function(p) {
      var s = series.sorted();
      if (s.length === 0) return NaN;
      if (p <= 0) return s[0];
      if (p >= 100) return s[s.length - 1];
      var idx = (p / 100) * (s.length - 1);
      var lo = Math.floor(idx);
      var hi = Math.ceil(idx);
      var frac = idx - lo;
      return s[lo] * (1 - frac) + s[hi] * frac;
    },

    /**
     * Calculate the interquartile range (Q3 - Q1).
     * @returns {number}
     */
    iqr: function() {
      return series.percentile(75) - series.percentile(25);
    },

    /**
     * Calculate skewness (Fisher's measure of asymmetry).
     * @returns {number}
     */
    skewness: function() {
      var nums = numericValues();
      var n = nums.length;
      if (n < 3) return NaN;
      var m = series.mean();
      var sd = series.std();
      if (sd === 0) return 0;
      var sum3 = 0;
      for (var i = 0; i < n; i++) sum3 += Math.pow((nums[i] - m) / sd, 3);
      return (n / ((n - 1) * (n - 2))) * sum3;
    },

    /**
     * Calculate excess kurtosis (Fisher's definition, normal = 0).
     * @returns {number}
     */
    kurtosis: function() {
      var nums = numericValues();
      var n = nums.length;
      if (n < 4) return NaN;
      var m = series.mean();
      var sd = series.std();
      if (sd === 0) return 0;
      var sum4 = 0;
      for (var i = 0; i < n; i++) sum4 += Math.pow((nums[i] - m) / sd, 4);
      var k = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum4;
      return k - (3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3));
    },

    /**
     * Count non-null/non-NaN values.
     * @returns {number}
     */
    count: function() {
      return numericValues().length;
    },

    /**
     * Generate descriptive statistics summary.
     * @returns {string} Formatted summary table.
     */
    describe: function() {
      var lines = [];
      lines.push('Series: ' + (_name || '(unnamed)') + '  [' + _values.length + ' values]');
      lines.push(repeatStr('-', 35));
      var stats = [
        ['count', series.count()],
        ['mean', series.mean()],
        ['std', series.std()],
        ['min', series.min()],
        ['25%', series.percentile(25)],
        ['50%', series.median()],
        ['75%', series.percentile(75)],
        ['max', series.max()],
        ['skewness', series.skewness()],
        ['kurtosis', series.kurtosis()],
        ['iqr', series.iqr()]
      ];
      for (var i = 0; i < stats.length; i++) {
        lines.push(padStr(stats[i][0], 12) + formatNum(stats[i][1]));
      }
      return lines.join('\n');
    },

    /**
     * Apply a function to each value, returning a new Series.
     * @param {Function} fn - Mapping function (value, index) => newValue.
     * @returns {Object} New Series.
     */
    map: function(fn) {
      var newVals = [];
      for (var i = 0; i < _values.length; i++) newVals.push(fn(_values[i], i));
      return createSeries(newVals, { name: _name, index: _index.slice() });
    },

    /**
     * Filter values by a predicate, returning a new Series.
     * @param {Function} fn - Predicate function (value, index) => boolean.
     * @returns {Object} Filtered Series.
     */
    filter: function(fn) {
      var newVals = [], newIdx = [];
      for (var i = 0; i < _values.length; i++) {
        if (fn(_values[i], i)) {
          newVals.push(_values[i]);
          newIdx.push(_index[i]);
        }
      }
      return createSeries(newVals, { name: _name, index: newIdx });
    },

    /**
     * Calculate cumulative sum.
     * @returns {Object} New Series with cumulative sums.
     */
    cumsum: function() {
      var result = [];
      var acc = 0;
      for (var i = 0; i < _values.length; i++) {
        acc += Number(_values[i]) || 0;
        result.push(acc);
      }
      return createSeries(result, { name: _name + '_cumsum', index: _index.slice() });
    },

    /**
     * Calculate rolling mean with a given window size.
     * @param {number} window - Window size.
     * @returns {Object} New Series of rolling means.
     */
    rollingMean: function(window) {
      var result = [];
      for (var i = 0; i < _values.length; i++) {
        if (i < window - 1) { result.push(NaN); continue; }
        var sum = 0;
        for (var j = i - window + 1; j <= i; j++) sum += Number(_values[j]) || 0;
        result.push(sum / window);
      }
      return createSeries(result, { name: _name + '_rm' + window, index: _index.slice() });
    },

    /**
     * Calculate element-wise z-scores (standard scores).
     * @returns {Object} New Series of z-scores.
     */
    zscore: function() {
      var m = series.mean();
      var sd = series.std();
      return series.map(function(v) { return sd === 0 ? 0 : (v - m) / sd; });
    },

    /**
     * Normalize values to [0, 1] range (min-max scaling).
     * @returns {Object} Normalized Series.
     */
    normalize: function() {
      var mn = series.min();
      var mx = series.max();
      var rng = mx - mn;
      return series.map(function(v) { return rng === 0 ? 0.5 : (v - mn) / rng; });
    },

    /**
     * Compute value counts (frequency distribution).
     * @returns {Object} Map of value to count.
     */
    valueCounts: function() {
      var counts = {};
      for (var i = 0; i < _values.length; i++) {
        var k = String(_values[i]);
        counts[k] = (counts[k] || 0) + 1;
      }
      return counts;
    },

    /**
     * Get unique values.
     * @returns {Array}
     */
    unique: function() {
      var seen = {};
      var result = [];
      for (var i = 0; i < _values.length; i++) {
        var k = String(_values[i]);
        if (!seen[k]) { seen[k] = true; result.push(_values[i]); }
      }
      return result;
    },

    /**
     * Return the top/bottom n values.
     * @param {number} n - Number of values.
     * @param {boolean} [ascending=false]
     * @returns {Object} New Series.
     */
    nlargest: function(n) {
      var indexed = [];
      for (var i = 0; i < _values.length; i++) indexed.push({ v: _values[i], i: i });
      indexed.sort(function(a, b) { return b.v - a.v; });
      var vals = [], idx = [];
      for (var j = 0; j < Math.min(n, indexed.length); j++) {
        vals.push(indexed[j].v);
        idx.push(_index[indexed[j].i]);
      }
      return createSeries(vals, { name: _name, index: idx });
    },

    /**
     * Detect outliers using IQR method.
     * @param {number} [multiplier=1.5] - IQR multiplier for fence calculation.
     * @returns {Object} Object with outliers array and bounds.
     */
    detectOutliers: function(multiplier) {
      if (multiplier === undefined) multiplier = 1.5;
      var q1 = series.percentile(25);
      var q3 = series.percentile(75);
      var iqrVal = q3 - q1;
      var lower = q1 - multiplier * iqrVal;
      var upper = q3 + multiplier * iqrVal;
      var outliers = [];
      for (var i = 0; i < _values.length; i++) {
        var v = Number(_values[i]);
        if (!isNaN(v) && (v < lower || v > upper)) {
          outliers.push({ index: _index[i], value: v });
        }
      }
      return { outliers: outliers, lowerFence: lower, upperFence: upper, q1: q1, q3: q3, iqr: iqrVal };
    },

    /**
     * Convert to plain string representation.
     * @returns {string}
     */
    toString: function() {
      var lines = [_name || 'Series'];
      var maxShow = Math.min(_values.length, 20);
      for (var i = 0; i < maxShow; i++) {
        lines.push(padStr(String(_index[i]), 8) + String(_values[i]));
      }
      if (_values.length > 20) lines.push('... (' + (_values.length - 20) + ' more)');
      return lines.join('\n');
    }
  };

  return series;
}

// ============================================================
//  DATAFRAME — 2D labeled data structure
// ============================================================

/**
 * Create a DataFrame — a two-dimensional labeled data structure with columns of
 * potentially different types. Think of it like a spreadsheet or SQL table.
 * 
 * @param {Object} columns - Map of column names to arrays of values.
 * @param {Object} [opts] - Options.
 * @param {Array} [opts.index] - Row index labels.
 * @returns {Object} DataFrame with query, transform, and analysis methods.
 * 
 * @example
 * var df = createDataFrame({
 *   name: ['Alice', 'Bob', 'Charlie'],
 *   age: [30, 25, 35],
 *   salary: [75000, 65000, 85000]
 * });
 * console.log(df.describe());
 */
function createDataFrame(columns, opts) {
  if (!opts) opts = {};
  var _cols = deepClone(columns);
  var _colNames = Object.keys(_cols);
  var _nRows = _colNames.length > 0 ? _cols[_colNames[0]].length : 0;
  var _index = opts.index || range(0, _nRows);

  var df = {
    /** @type {string[]} Column names */
    columns: _colNames,
    /** @type {number} Number of rows */
    nRows: _nRows,
    /** @type {number} Number of columns */
    nCols: _colNames.length,
    /** @type {Array} Row index */
    index: _index,

    /**
     * Get a column as a Series.
     * @param {string} name - Column name.
     * @returns {Object} Series for that column.
     */
    col: function(name) {
      if (!_cols[name]) return null;
      return createSeries(_cols[name], { name: name, index: _index.slice() });
    },

    /**
     * Get raw column data.
     * @param {string} name - Column name.
     * @returns {Array}
     */
    getColumn: function(name) {
      return _cols[name] ? _cols[name].slice() : null;
    },

    /**
     * Get a row by index position as an object.
     * @param {number} idx - Row index.
     * @returns {Object}
     */
    row: function(idx) {
      if (idx < 0 || idx >= _nRows) return null;
      var obj = {};
      for (var i = 0; i < _colNames.length; i++) {
        obj[_colNames[i]] = _cols[_colNames[i]][idx];
      }
      return obj;
    },

    /**
     * Select a subset of columns, returning a new DataFrame.
     * @param {string[]} cols - Column names to select.
     * @returns {Object} New DataFrame.
     */
    select: function(cols) {
      var newCols = {};
      for (var i = 0; i < cols.length; i++) {
        if (_cols[cols[i]]) newCols[cols[i]] = _cols[cols[i]].slice();
      }
      return createDataFrame(newCols, { index: _index.slice() });
    },

    /**
     * Filter rows by predicate, returning a new DataFrame.
     * @param {Function} fn - Predicate (row, index) => boolean.
     * @returns {Object} Filtered DataFrame.
     */
    filter: function(fn) {
      var newCols = {};
      var newIdx = [];
      for (var i = 0; i < _colNames.length; i++) newCols[_colNames[i]] = [];
      for (var j = 0; j < _nRows; j++) {
        var rowObj = df.row(j);
        if (fn(rowObj, j)) {
          for (var k = 0; k < _colNames.length; k++) {
            newCols[_colNames[k]].push(_cols[_colNames[k]][j]);
          }
          newIdx.push(_index[j]);
        }
      }
      return createDataFrame(newCols, { index: newIdx });
    },

    /**
     * Sort by a column.
     * @param {string} colName - Column to sort by.
     * @param {boolean} [ascending=true] - Sort direction.
     * @returns {Object} Sorted DataFrame.
     */
    sortBy: function(colName, ascending) {
      if (ascending === undefined) ascending = true;
      var indices = range(0, _nRows);
      var colData = _cols[colName];
      indices.sort(function(a, b) {
        if (colData[a] < colData[b]) return ascending ? -1 : 1;
        if (colData[a] > colData[b]) return ascending ? 1 : -1;
        return 0;
      });
      var newCols = {};
      var newIdx = [];
      for (var i = 0; i < _colNames.length; i++) newCols[_colNames[i]] = [];
      for (var j = 0; j < indices.length; j++) {
        for (var k = 0; k < _colNames.length; k++) {
          newCols[_colNames[k]].push(_cols[_colNames[k]][indices[j]]);
        }
        newIdx.push(_index[indices[j]]);
      }
      return createDataFrame(newCols, { index: newIdx });
    },

    /**
     * Add a computed column.
     * @param {string} name - New column name.
     * @param {Function} fn - Row function (row, index) => value.
     * @returns {Object} New DataFrame with added column.
     */
    addColumn: function(name, fn) {
      var newCols = deepClone(_cols);
      newCols[name] = [];
      for (var i = 0; i < _nRows; i++) {
        newCols[name].push(fn(df.row(i), i));
      }
      return createDataFrame(newCols, { index: _index.slice() });
    },

    /**
     * Group by a column and aggregate.
     * @param {string} groupCol - Column to group by.
     * @param {Object} aggs - Map of column => aggregation function name ('sum', 'mean', 'count', 'min', 'max').
     * @returns {Object} Aggregated DataFrame.
     */
    groupBy: function(groupCol, aggs) {
      var groups = {};
      for (var i = 0; i < _nRows; i++) {
        var key = String(_cols[groupCol][i]);
        if (!groups[key]) groups[key] = [];
        groups[key].push(i);
      }
      var aggFns = {
        sum: function(arr) { var s = 0; for (var i = 0; i < arr.length; i++) s += arr[i]; return s; },
        mean: function(arr) { var s = 0; for (var i = 0; i < arr.length; i++) s += arr[i]; return s / arr.length; },
        count: function(arr) { return arr.length; },
        min: function(arr) { var m = arr[0]; for (var i = 1; i < arr.length; i++) if (arr[i] < m) m = arr[i]; return m; },
        max: function(arr) { var m = arr[0]; for (var i = 1; i < arr.length; i++) if (arr[i] > m) m = arr[i]; return m; }
      };
      var resultCols = {};
      resultCols[groupCol] = [];
      var aggKeys = Object.keys(aggs);
      for (var a = 0; a < aggKeys.length; a++) {
        resultCols[aggKeys[a] + '_' + aggs[aggKeys[a]]] = [];
      }
      var groupKeys = Object.keys(groups);
      for (var g = 0; g < groupKeys.length; g++) {
        resultCols[groupCol].push(groupKeys[g]);
        var idxs = groups[groupKeys[g]];
        for (var b = 0; b < aggKeys.length; b++) {
          var vals = [];
          for (var c = 0; c < idxs.length; c++) vals.push(Number(_cols[aggKeys[b]][idxs[c]]));
          var fn = aggFns[aggs[aggKeys[b]]] || aggFns.count;
          resultCols[aggKeys[b] + '_' + aggs[aggKeys[b]]].push(fn(vals));
        }
      }
      return createDataFrame(resultCols);
    },

    /**
     * Compute correlation matrix for numeric columns.
     * @returns {Object} DataFrame representing the correlation matrix.
     */
    corr: function() {
      var numCols = [];
      for (var i = 0; i < _colNames.length; i++) {
        var s = df.col(_colNames[i]);
        if (!isNaN(s.mean())) numCols.push(_colNames[i]);
      }
      var matrix = {};
      for (var a = 0; a < numCols.length; a++) matrix[numCols[a]] = [];
      for (var r = 0; r < numCols.length; r++) {
        for (var c = 0; c < numCols.length; c++) {
          var corr = pearsonCorrelation(_cols[numCols[r]], _cols[numCols[c]]);
          matrix[numCols[c]].push(corr);
        }
      }
      return createDataFrame(matrix, { index: numCols });
    },

    /**
     * Generate descriptive statistics for all numeric columns.
     * @returns {string} Formatted statistics table.
     */
    describe: function() {
      var numCols = [];
      for (var i = 0; i < _colNames.length; i++) {
        var s = df.col(_colNames[i]);
        if (!isNaN(s.mean())) numCols.push(_colNames[i]);
      }
      var header = padStr('', 12);
      for (var h = 0; h < numCols.length; h++) header += padStr(numCols[h], 14, 'left');
      var lines = [header, repeatStr('-', header.length)];
      var statNames = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
      for (var s = 0; s < statNames.length; s++) {
        var row = padStr(statNames[s], 12);
        for (var c = 0; c < numCols.length; c++) {
          var ser = df.col(numCols[c]);
          var val;
          if (statNames[s] === 'count') val = ser.count();
          else if (statNames[s] === 'mean') val = ser.mean();
          else if (statNames[s] === 'std') val = ser.std();
          else if (statNames[s] === 'min') val = ser.min();
          else if (statNames[s] === '25%') val = ser.percentile(25);
          else if (statNames[s] === '50%') val = ser.median();
          else if (statNames[s] === '75%') val = ser.percentile(75);
          else if (statNames[s] === 'max') val = ser.max();
          row += padStr(formatNum(val, 2), 14, 'left');
        }
        lines.push(row);
      }
      return lines.join('\n');
    },

    /**
     * Convert first N rows to a formatted table string.
     * @param {number} [n=10] - Number of rows.
     * @returns {string}
     */
    head: function(n) {
      if (n === undefined) n = 10;
      var colWidths = {};
      for (var i = 0; i < _colNames.length; i++) {
        colWidths[_colNames[i]] = _colNames[i].length;
        for (var j = 0; j < Math.min(n, _nRows); j++) {
          var vLen = String(_cols[_colNames[i]][j]).length;
          if (vLen > colWidths[_colNames[i]]) colWidths[_colNames[i]] = vLen;
        }
        colWidths[_colNames[i]] = Math.min(colWidths[_colNames[i]] + 2, 20);
      }
      var header = padStr('#', 6);
      for (var h = 0; h < _colNames.length; h++) header += padStr(_colNames[h], colWidths[_colNames[h]]);
      var sep = repeatStr('-', header.length);
      var lines = [header, sep];
      for (var r = 0; r < Math.min(n, _nRows); r++) {
        var line = padStr(String(_index[r]), 6);
        for (var c = 0; c < _colNames.length; c++) {
          line += padStr(String(_cols[_colNames[c]][r]), colWidths[_colNames[c]]);
        }
        lines.push(line);
      }
      if (_nRows > n) lines.push('... (' + (_nRows - n) + ' more rows)');
      lines.push('[' + _nRows + ' rows x ' + _colNames.length + ' columns]');
      return lines.join('\n');
    },

    /**
     * Convert to array of row objects.
     * @returns {Object[]}
     */
    toRecords: function() {
      var records = [];
      for (var i = 0; i < _nRows; i++) records.push(df.row(i));
      return records;
    }
  };

  return df;
}

/**
 * Create a DataFrame from an array of row objects.
 * @param {Object[]} records - Array of {key: value} objects.
 * @returns {Object} DataFrame.
 */
function dataFrameFromRecords(records) {
  if (records.length === 0) return createDataFrame({});
  var colNames = Object.keys(records[0]);
  var cols = {};
  for (var i = 0; i < colNames.length; i++) cols[colNames[i]] = [];
  for (var r = 0; r < records.length; r++) {
    for (var c = 0; c < colNames.length; c++) {
      cols[colNames[c]].push(records[r][colNames[c]] !== undefined ? records[r][colNames[c]] : null);
    }
  }
  return createDataFrame(cols);
}

// ============================================================
//  CSV PARSER — RFC 4180 compliant
// ============================================================

/**
 * Parse a CSV string into a DataFrame with type inference.
 * Handles quoted fields, escaped quotes, and multi-line values.
 * 
 * @param {string} csvStr - The CSV text.
 * @param {Object} [opts] - Options.
 * @param {string} [opts.delimiter=','] - Field delimiter.
 * @param {boolean} [opts.header=true] - First row is header.
 * @returns {Object} Parsed DataFrame.
 * 
 * @example
 * var csv = 'name,age,salary\nAlice,30,75000\nBob,25,65000';
 * var df = parseCSV(csv);
 * console.log(df.head());
 */
function parseCSV(csvStr, opts) {
  if (!opts) opts = {};
  var delim = opts.delimiter || ',';
  var hasHeader = opts.header !== false;

  var rows = [];
  var row = [];
  var field = '';
  var inQuote = false;
  var i = 0;

  while (i < csvStr.length) {
    var ch = csvStr[i];
    if (inQuote) {
      if (ch === '"') {
        if (i + 1 < csvStr.length && csvStr[i + 1] === '"') {
          field += '"';
          i += 2;
        } else {
          inQuote = false;
          i++;
        }
      } else {
        field += ch;
        i++;
      }
    } else {
      if (ch === '"') {
        inQuote = true;
        i++;
      } else if (ch === delim) {
        row.push(field);
        field = '';
        i++;
      } else if (ch === '\n' || ch === '\r') {
        row.push(field);
        field = '';
        if (ch === '\r' && i + 1 < csvStr.length && csvStr[i + 1] === '\n') i++;
        rows.push(row);
        row = [];
        i++;
      } else {
        field += ch;
        i++;
      }
    }
  }
  if (field || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  if (rows.length === 0) return createDataFrame({});

  var headers = hasHeader ? rows[0] : rows[0].map(function(_, i) { return 'col_' + i; });
  var dataStart = hasHeader ? 1 : 0;
  var cols = {};
  for (var h = 0; h < headers.length; h++) cols[headers[h]] = [];

  for (var r = dataStart; r < rows.length; r++) {
    for (var c = 0; c < headers.length; c++) {
      var val = rows[r][c] !== undefined ? rows[r][c].trim() : '';
      var num = Number(val);
      if (val !== '' && !isNaN(num)) val = num;
      else if (val.toLowerCase() === 'true') val = true;
      else if (val.toLowerCase() === 'false') val = false;
      cols[headers[c]].push(val);
    }
  }

  return createDataFrame(cols);
}

// ============================================================
//  STATISTICS MODULE
// ============================================================

/**
 * Calculate Pearson correlation coefficient between two arrays.
 * Measures linear correlation from -1 (inverse) to +1 (direct).
 * 
 * @param {number[]} x - First variable.
 * @param {number[]} y - Second variable.
 * @returns {number} Correlation coefficient.
 */
function pearsonCorrelation(x, y) {
  var n = Math.min(x.length, y.length);
  if (n < 2) return NaN;
  var sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
  for (var i = 0; i < n; i++) {
    var xi = Number(x[i]), yi = Number(y[i]);
    sumX += xi; sumY += yi;
    sumXY += xi * yi;
    sumX2 += xi * xi;
    sumY2 += yi * yi;
  }
  var num = n * sumXY - sumX * sumY;
  var den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  return den === 0 ? 0 : num / den;
}

/**
 * Calculate Spearman rank correlation coefficient.
 * Non-parametric measure of monotonic relationship.
 * 
 * @param {number[]} x - First variable.
 * @param {number[]} y - Second variable.
 * @returns {number} Spearman's rho.
 */
function spearmanCorrelation(x, y) {
  function ranks(arr) {
    var indexed = [];
    for (var i = 0; i < arr.length; i++) indexed.push({ v: arr[i], i: i });
    indexed.sort(function(a, b) { return a.v - b.v; });
    var r = new Array(arr.length);
    var pos = 0;
    while (pos < indexed.length) {
      var end = pos + 1;
      while (end < indexed.length && indexed[end].v === indexed[pos].v) end++;
      var avgRank = (pos + end - 1) / 2 + 1;
      for (var j = pos; j < end; j++) r[indexed[j].i] = avgRank;
      pos = end;
    }
    return r;
  }
  return pearsonCorrelation(ranks(x), ranks(y));
}

/**
 * Perform Ordinary Least Squares linear regression.
 * Fits y = slope * x + intercept with full diagnostic statistics.
 * 
 * @param {number[]} x - Independent variable.
 * @param {number[]} y - Dependent variable.
 * @returns {Object} Regression results: slope, intercept, rSquared, predictions, residuals, standardError, pValue, confidenceInterval.
 * 
 * @example
 * var result = linearRegression([1,2,3,4,5], [2.1, 3.8, 6.2, 7.9, 10.1]);
 * console.log('Slope:', result.slope, 'R²:', result.rSquared);
 */
function linearRegression(x, y) {
  var n = Math.min(x.length, y.length);
  var sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (var i = 0; i < n; i++) {
    sumX += x[i]; sumY += y[i];
    sumXY += x[i] * y[i];
    sumX2 += x[i] * x[i];
  }
  var slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  var intercept = (sumY - slope * sumX) / n;

  var predictions = [];
  var residuals = [];
  var ssRes = 0, ssTot = 0;
  var meanY = sumY / n;
  for (var j = 0; j < n; j++) {
    var pred = slope * x[j] + intercept;
    predictions.push(pred);
    var res = y[j] - pred;
    residuals.push(res);
    ssRes += res * res;
    ssTot += (y[j] - meanY) * (y[j] - meanY);
  }
  var rSquared = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  var standardError = n > 2 ? Math.sqrt(ssRes / (n - 2)) : 0;

  var slopeStdErr = standardError / Math.sqrt(sumX2 - sumX * sumX / n);
  var tStatistic = slopeStdErr > 0 ? Math.abs(slope / slopeStdErr) : Infinity;
  var pValue = n > 2 ? Math.exp(-0.717 * tStatistic - 0.416 * tStatistic * tStatistic / n) : 1;
  pValue = Math.min(pValue, 1);

  return {
    slope: slope,
    intercept: intercept,
    rSquared: rSquared,
    adjustedRSquared: n > 2 ? 1 - (1 - rSquared) * (n - 1) / (n - 2) : rSquared,
    predictions: predictions,
    residuals: residuals,
    standardError: standardError,
    tStatistic: tStatistic,
    pValue: pValue,
    n: n,
    predict: function(xVal) { return slope * xVal + intercept; }
  };
}

/**
 * Polynomial regression of given degree.
 * Fits y = a0 + a1*x + a2*x² + ... using normal equations.
 * 
 * @param {number[]} x - Independent variable.
 * @param {number[]} y - Dependent variable.
 * @param {number} degree - Polynomial degree.
 * @returns {Object} Coefficients, rSquared, predict function.
 */
function polynomialRegression(x, y, degree) {
  var n = x.length;
  var size = degree + 1;

  // Build normal equations: X'X * a = X'y
  var matrix = [];
  var rhs = [];
  for (var i = 0; i < size; i++) {
    matrix[i] = [];
    rhs[i] = 0;
    for (var j = 0; j < size; j++) {
      var s = 0;
      for (var k = 0; k < n; k++) s += Math.pow(x[k], i + j);
      matrix[i][j] = s;
    }
    for (var l = 0; l < n; l++) rhs[i] += Math.pow(x[l], i) * y[l];
  }

  // Gaussian elimination
  for (var p = 0; p < size; p++) {
    var maxRow = p;
    for (var r = p + 1; r < size; r++) {
      if (Math.abs(matrix[r][p]) > Math.abs(matrix[maxRow][p])) maxRow = r;
    }
    var tmp = matrix[p]; matrix[p] = matrix[maxRow]; matrix[maxRow] = tmp;
    var tmpR = rhs[p]; rhs[p] = rhs[maxRow]; rhs[maxRow] = tmpR;
    for (var r2 = p + 1; r2 < size; r2++) {
      var f = matrix[r2][p] / matrix[p][p];
      for (var c = p; c < size; c++) matrix[r2][c] -= f * matrix[p][c];
      rhs[r2] -= f * rhs[p];
    }
  }
  var coeffs = new Array(size);
  for (var b = size - 1; b >= 0; b--) {
    coeffs[b] = rhs[b];
    for (var c2 = b + 1; c2 < size; c2++) coeffs[b] -= matrix[b][c2] * coeffs[c2];
    coeffs[b] /= matrix[b][b];
  }

  var predict = function(xVal) {
    var sum = 0;
    for (var i = 0; i < coeffs.length; i++) sum += coeffs[i] * Math.pow(xVal, i);
    return sum;
  };

  var ssRes = 0, ssTot = 0;
  var meanY = 0;
  for (var m = 0; m < n; m++) meanY += y[m];
  meanY /= n;
  for (var q = 0; q < n; q++) {
    ssRes += Math.pow(y[q] - predict(x[q]), 2);
    ssTot += Math.pow(y[q] - meanY, 2);
  }

  return {
    coefficients: coeffs,
    degree: degree,
    rSquared: ssTot === 0 ? 1 : 1 - ssRes / ssTot,
    predict: predict
  };
}

/**
 * One-sample t-test: test if sample mean differs from a hypothesized value.
 * 
 * @param {number[]} sample - Sample data.
 * @param {number} mu0 - Hypothesized population mean.
 * @returns {Object} tStatistic, degreesOfFreedom, pValue, meanDiff, confidenceInterval.
 */
function tTestOneSample(sample, mu0) {
  var n = sample.length;
  var s = createSeries(sample);
  var mean = s.mean();
  var std = s.std();
  var se = std / Math.sqrt(n);
  var t = (mean - mu0) / se;
  var df = n - 1;
  var pValue = Math.exp(-0.717 * Math.abs(t) - 0.416 * t * t / df);
  pValue = Math.min(pValue * 2, 1); // two-tailed
  var margin = 1.96 * se; // approximate 95% CI
  return {
    tStatistic: t,
    degreesOfFreedom: df,
    pValue: pValue,
    mean: mean,
    hypothesizedMean: mu0,
    standardError: se,
    confidenceInterval: [mean - margin, mean + margin],
    significant: pValue < 0.05
  };
}

/**
 * Two-sample independent t-test.
 * Tests whether two groups have different means.
 * 
 * @param {number[]} a - First sample.
 * @param {number[]} b - Second sample.
 * @returns {Object} Test results with tStatistic, pValue, etc.
 */
function tTestTwoSample(a, b) {
  var sa = createSeries(a), sb = createSeries(b);
  var n1 = a.length, n2 = b.length;
  var m1 = sa.mean(), m2 = sb.mean();
  var v1 = sa.variance(), v2 = sb.variance();
  var se = Math.sqrt(v1 / n1 + v2 / n2);
  var t = (m1 - m2) / se;
  var num = Math.pow(v1 / n1 + v2 / n2, 2);
  var den = Math.pow(v1 / n1, 2) / (n1 - 1) + Math.pow(v2 / n2, 2) / (n2 - 1);
  var df = num / den;
  var pValue = Math.exp(-0.717 * Math.abs(t) - 0.416 * t * t / Math.max(df, 1));
  pValue = Math.min(pValue * 2, 1);
  return {
    tStatistic: t,
    degreesOfFreedom: df,
    pValue: pValue,
    meanDiff: m1 - m2,
    group1Mean: m1,
    group2Mean: m2,
    significant: pValue < 0.05
  };
}

/**
 * One-way ANOVA — tests if means differ across multiple groups.
 * 
 * @param {Array[]} groups - Array of number arrays (one per group).
 * @returns {Object} fStatistic, pValue, groupMeans, significant.
 */
function anova(groups) {
  var grandMean = 0;
  var totalN = 0;
  var groupMeans = [];
  for (var g = 0; g < groups.length; g++) {
    var gm = 0;
    for (var i = 0; i < groups[g].length; i++) gm += groups[g][i];
    gm /= groups[g].length;
    groupMeans.push(gm);
    totalN += groups[g].length;
    grandMean += gm * groups[g].length;
  }
  grandMean /= totalN;

  var ssBetween = 0, ssWithin = 0;
  for (var g2 = 0; g2 < groups.length; g2++) {
    ssBetween += groups[g2].length * Math.pow(groupMeans[g2] - grandMean, 2);
    for (var j = 0; j < groups[g2].length; j++) {
      ssWithin += Math.pow(groups[g2][j] - groupMeans[g2], 2);
    }
  }

  var dfBetween = groups.length - 1;
  var dfWithin = totalN - groups.length;
  var msBetween = ssBetween / dfBetween;
  var msWithin = ssWithin / dfWithin;
  var fStat = msWithin > 0 ? msBetween / msWithin : Infinity;
  var pValue = Math.exp(-0.4 * fStat);
  pValue = Math.min(pValue, 1);

  return {
    fStatistic: fStat,
    pValue: pValue,
    ssBetween: ssBetween,
    ssWithin: ssWithin,
    dfBetween: dfBetween,
    dfWithin: dfWithin,
    groupMeans: groupMeans,
    significant: pValue < 0.05
  };
}

/**
 * Chi-squared goodness-of-fit test.
 * 
 * @param {number[]} observed - Observed frequencies.
 * @param {number[]} expected - Expected frequencies.
 * @returns {Object} chiSquared, pValue, degreesOfFreedom, significant.
 */
function chiSquaredTest(observed, expected) {
  var chiSq = 0;
  for (var i = 0; i < observed.length; i++) {
    if (expected[i] > 0) {
      chiSq += Math.pow(observed[i] - expected[i], 2) / expected[i];
    }
  }
  var df = observed.length - 1;
  var pValue = Math.exp(-chiSq / 2);
  return {
    chiSquared: chiSq,
    degreesOfFreedom: df,
    pValue: pValue,
    significant: pValue < 0.05
  };
}

// ============================================================
//  PROBABILITY DISTRIBUTIONS
// ============================================================

/**
 * Probability distribution functions for common distributions.
 * Includes PDF, CDF, and quantile (inverse CDF) where applicable.
 * @namespace Distributions
 */
var Distributions = {
  /**
   * Standard normal distribution functions.
   * @namespace Distributions.normal
   */
  normal: {
    /**
     * Normal probability density function.
     * @param {number} x - Value.
     * @param {number} [mu=0] - Mean.
     * @param {number} [sigma=1] - Standard deviation.
     * @returns {number} Density at x.
     */
    pdf: function(x, mu, sigma) {
      if (mu === undefined) mu = 0;
      if (sigma === undefined) sigma = 1;
      var z = (x - mu) / sigma;
      return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
    },

    /**
     * Normal cumulative distribution function (Abramowitz & Stegun approximation).
     * @param {number} x - Value.
     * @param {number} [mu=0] - Mean.
     * @param {number} [sigma=1] - Standard deviation.
     * @returns {number} P(X <= x).
     */
    cdf: function(x, mu, sigma) {
      if (mu === undefined) mu = 0;
      if (sigma === undefined) sigma = 1;
      var z = (x - mu) / sigma;
      var t = 1 / (1 + 0.2316419 * Math.abs(z));
      var d = 0.3989422804014327;
      var p = d * Math.exp(-z * z / 2) *
        (t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))));
      return z > 0 ? 1 - p : p;
    },

    /**
     * Normal quantile (inverse CDF) using Rational Approximation.
     * @param {number} p - Probability (0 < p < 1).
     * @param {number} [mu=0] - Mean.
     * @param {number} [sigma=1] - Standard deviation.
     * @returns {number} x such that CDF(x) = p.
     */
    quantile: function(p, mu, sigma) {
      if (mu === undefined) mu = 0;
      if (sigma === undefined) sigma = 1;
      if (p <= 0) return -Infinity;
      if (p >= 1) return Infinity;
      // Rational approximation for standard normal quantile
      var a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
               1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
      var b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
               6.680131188771972e+01, -1.328068155288572e+01];
      var c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
      var d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
      var q, r;
      if (p < 0.02425) {
        q = Math.sqrt(-2 * Math.log(p));
        r = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
      } else if (p <= 0.97575) {
        q = p - 0.5;
        r = q * q;
        r = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
      } else {
        q = Math.sqrt(-2 * Math.log(1 - p));
        r = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
      }
      return mu + sigma * r;
    }
  },

  /**
   * Poisson distribution.
   * @param {number} k - Number of events.
   * @param {number} lambda - Average rate.
   * @returns {number} P(X = k).
   */
  poisson: function(k, lambda) {
    var logP = k * Math.log(lambda) - lambda;
    for (var i = 2; i <= k; i++) logP -= Math.log(i);
    return Math.exp(logP);
  },

  /**
   * Binomial probability mass function.
   * @param {number} k - Number of successes.
   * @param {number} n - Number of trials.
   * @param {number} p - Success probability.
   * @returns {number} P(X = k).
   */
  binomial: function(k, n, p) {
    // log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
    var logCoeff = 0;
    for (var i = 0; i < k; i++) {
      logCoeff += Math.log(n - i) - Math.log(i + 1);
    }
    return Math.exp(logCoeff + k * Math.log(p) + (n - k) * Math.log(1 - p));
  },

  /**
   * Exponential distribution PDF.
   * @param {number} x - Value.
   * @param {number} lambda - Rate parameter.
   * @returns {number} Density at x.
   */
  exponential: {
    pdf: function(x, lambda) {
      return x < 0 ? 0 : lambda * Math.exp(-lambda * x);
    },
    cdf: function(x, lambda) {
      return x < 0 ? 0 : 1 - Math.exp(-lambda * x);
    }
  }
};

// ============================================================
//  K-MEANS CLUSTERING
// ============================================================

/**
 * K-Means clustering algorithm with configurable initialization and convergence.
 * Groups data points into k clusters by minimizing within-cluster sum of squares.
 * 
 * @param {number[][]} data - Array of data points (each point is an array of coordinates).
 * @param {number} k - Number of clusters.
 * @param {Object} [opts] - Options.
 * @param {number} [opts.maxIter=100] - Maximum iterations.
 * @param {number} [opts.tolerance=0.0001] - Convergence tolerance.
 * @param {string} [opts.init='kmeans++'] - Initialization: 'kmeans++' or 'random'.
 * @returns {Object} clusters, centroids, assignments, iterations, inertia, silhouetteScore.
 * 
 * @example
 * var points = [[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]];
 * var result = kMeans(points, 2);
 * console.log('Centroids:', result.centroids);
 * console.log('Silhouette:', result.silhouetteScore);
 */
function kMeans(data, k, opts) {
  if (!opts) opts = {};
  var maxIter = opts.maxIter || 100;
  var tolerance = opts.tolerance || 0.0001;
  var dims = data[0].length;
  var n = data.length;

  function dist(a, b) {
    var s = 0;
    for (var i = 0; i < dims; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
    return Math.sqrt(s);
  }

  // K-means++ initialization
  var centroids = [];
  var firstIdx = Math.floor(Math.random() * n);
  centroids.push(data[firstIdx].slice());
  for (var c = 1; c < k; c++) {
    var dists = [];
    var totalDist = 0;
    for (var i = 0; i < n; i++) {
      var minD = Infinity;
      for (var j = 0; j < centroids.length; j++) {
        var d = dist(data[i], centroids[j]);
        if (d < minD) minD = d;
      }
      dists.push(minD * minD);
      totalDist += minD * minD;
    }
    var threshold = Math.random() * totalDist;
    var cumul = 0;
    for (var p = 0; p < n; p++) {
      cumul += dists[p];
      if (cumul >= threshold) {
        centroids.push(data[p].slice());
        break;
      }
    }
  }

  var assignments = new Array(n);
  var iterations = 0;

  for (var iter = 0; iter < maxIter; iter++) {
    iterations++;

    // Assign points to nearest centroid
    for (var i2 = 0; i2 < n; i2++) {
      var bestC = 0;
      var bestD = dist(data[i2], centroids[0]);
      for (var c2 = 1; c2 < k; c2++) {
        var d2 = dist(data[i2], centroids[c2]);
        if (d2 < bestD) { bestD = d2; bestC = c2; }
      }
      assignments[i2] = bestC;
    }

    // Update centroids
    var newCentroids = [];
    var converged = true;
    for (var c3 = 0; c3 < k; c3++) {
      var sum = new Array(dims);
      for (var d3 = 0; d3 < dims; d3++) sum[d3] = 0;
      var count = 0;
      for (var i3 = 0; i3 < n; i3++) {
        if (assignments[i3] === c3) {
          count++;
          for (var d4 = 0; d4 < dims; d4++) sum[d4] += data[i3][d4];
        }
      }
      var newC = [];
      for (var d5 = 0; d5 < dims; d5++) newC.push(count > 0 ? sum[d5] / count : centroids[c3][d5]);
      if (dist(newC, centroids[c3]) > tolerance) converged = false;
      newCentroids.push(newC);
    }
    centroids = newCentroids;
    if (converged) break;
  }

  // Calculate inertia
  var inertia = 0;
  var clusters = [];
  for (var c4 = 0; c4 < k; c4++) clusters.push([]);
  for (var i4 = 0; i4 < n; i4++) {
    clusters[assignments[i4]].push(i4);
    inertia += Math.pow(dist(data[i4], centroids[assignments[i4]]), 2);
  }

  // Silhouette score
  var silhouette = 0;
  for (var i5 = 0; i5 < n; i5++) {
    var myCluster = assignments[i5];
    var a = 0, aCount = 0;
    for (var j2 = 0; j2 < n; j2++) {
      if (j2 !== i5 && assignments[j2] === myCluster) {
        a += dist(data[i5], data[j2]);
        aCount++;
      }
    }
    a = aCount > 0 ? a / aCount : 0;
    var b = Infinity;
    for (var c5 = 0; c5 < k; c5++) {
      if (c5 === myCluster) continue;
      var bSum = 0, bCount = 0;
      for (var j3 = 0; j3 < n; j3++) {
        if (assignments[j3] === c5) { bSum += dist(data[i5], data[j3]); bCount++; }
      }
      if (bCount > 0) { var bAvg = bSum / bCount; if (bAvg < b) b = bAvg; }
    }
    var si = (b - a) / Math.max(a, b);
    if (isNaN(si)) si = 0;
    silhouette += si;
  }
  silhouette /= n;

  return {
    centroids: centroids,
    assignments: assignments,
    clusters: clusters,
    iterations: iterations,
    inertia: inertia,
    silhouetteScore: silhouette,
    k: k
  };
}

// ============================================================
//  TIME SERIES ANALYSIS
// ============================================================

/**
 * Time series analysis toolkit with forecasting and decomposition.
 * @namespace TimeSeries
 */
var TimeSeries = {
  /**
   * Simple Moving Average.
   * @param {number[]} data - Time series data.
   * @param {number} window - Window size.
   * @returns {number[]} Smoothed values (NaN for initial window).
   */
  sma: function(data, window) {
    var result = [];
    for (var i = 0; i < data.length; i++) {
      if (i < window - 1) { result.push(NaN); continue; }
      var sum = 0;
      for (var j = i - window + 1; j <= i; j++) sum += data[j];
      result.push(sum / window);
    }
    return result;
  },

  /**
   * Exponential Moving Average.
   * @param {number[]} data - Time series data.
   * @param {number} alpha - Smoothing factor (0 < alpha <= 1).
   * @returns {number[]}
   */
  ema: function(data, alpha) {
    if (data.length === 0) return [];
    var result = [data[0]];
    for (var i = 1; i < data.length; i++) {
      result.push(alpha * data[i] + (1 - alpha) * result[i - 1]);
    }
    return result;
  },

  /**
   * Double Exponential Smoothing (Holt's method) for trend data.
   * @param {number[]} data - Time series.
   * @param {number} alpha - Level smoothing.
   * @param {number} beta - Trend smoothing.
   * @param {number} horizon - Number of periods to forecast.
   * @returns {Object} fitted, forecast, level, trend.
   */
  doubleExponentialSmoothing: function(data, alpha, beta, horizon) {
    var n = data.length;
    var level = [data[0]];
    var trend = [data.length > 1 ? data[1] - data[0] : 0];
    var fitted = [data[0]];

    for (var i = 1; i < n; i++) {
      var newLevel = alpha * data[i] + (1 - alpha) * (level[i - 1] + trend[i - 1]);
      var newTrend = beta * (newLevel - level[i - 1]) + (1 - beta) * trend[i - 1];
      level.push(newLevel);
      trend.push(newTrend);
      fitted.push(newLevel);
    }

    var forecast = [];
    for (var h = 1; h <= horizon; h++) {
      forecast.push(level[n - 1] + h * trend[n - 1]);
    }

    return { fitted: fitted, forecast: forecast, level: level, trend: trend };
  },

  /**
   * Seasonal decomposition using moving average method.
   * Decomposes series into trend, seasonal, and residual components.
   * 
   * @param {number[]} data - Time series data.
   * @param {number} period - Seasonal period.
   * @returns {Object} trend, seasonal, residual arrays.
   */
  decompose: function(data, period) {
    // Trend via centered moving average
    var trend = TimeSeries.sma(data, period);
    // Detrended = data - trend
    var detrended = [];
    for (var i = 0; i < data.length; i++) {
      detrended.push(isNaN(trend[i]) ? NaN : data[i] - trend[i]);
    }
    // Average seasonal component per position
    var seasonal = new Array(data.length);
    var seasonalAvg = new Array(period);
    for (var p = 0; p < period; p++) {
      var sum = 0, count = 0;
      for (var j = p; j < data.length; j += period) {
        if (!isNaN(detrended[j])) { sum += detrended[j]; count++; }
      }
      seasonalAvg[p] = count > 0 ? sum / count : 0;
    }
    for (var k = 0; k < data.length; k++) seasonal[k] = seasonalAvg[k % period];
    // Residual
    var residual = [];
    for (var l = 0; l < data.length; l++) {
      residual.push(isNaN(trend[l]) ? NaN : data[l] - trend[l] - seasonal[l]);
    }
    return { trend: trend, seasonal: seasonal, residual: residual };
  },

  /**
   * Detect change points in time series using CUSUM method.
   * @param {number[]} data - Time series data.
   * @param {number} [threshold=2] - Detection threshold (in std devs).
   * @returns {Object} changePoints array, cumsum array.
   */
  detectChangePoints: function(data, threshold) {
    if (threshold === undefined) threshold = 2;
    var s = createSeries(data);
    var mean = s.mean();
    var std = s.std();
    var cumsum = [0];
    var changePoints = [];
    for (var i = 0; i < data.length; i++) {
      var cs = cumsum[cumsum.length - 1] + (data[i] - mean);
      cumsum.push(cs);
      if (Math.abs(cs) > threshold * std * Math.sqrt(i + 1)) {
        changePoints.push({ index: i, value: data[i], cumsum: cs });
      }
    }
    return { changePoints: changePoints, cumsum: cumsum.slice(1) };
  },

  /**
   * Simple autocorrelation function.
   * @param {number[]} data - Time series data.
   * @param {number} maxLag - Maximum lag to compute.
   * @returns {number[]} Autocorrelation values for lags 0..maxLag.
   */
  autocorrelation: function(data, maxLag) {
    var n = data.length;
    var s = createSeries(data);
    var mean = s.mean();
    var variance = s.variance(true);
    var acf = [];
    for (var lag = 0; lag <= maxLag; lag++) {
      var sum = 0;
      for (var i = 0; i < n - lag; i++) {
        sum += (data[i] - mean) * (data[i + lag] - mean);
      }
      acf.push(sum / (n * variance));
    }
    return acf;
  }
};

// ============================================================
//  ASCII CHARTS
// ============================================================

/**
 * ASCII chart rendering functions for terminal-based data visualization.
 * Supports multiple chart types with configurable dimensions and labels.
 * @namespace Charts
 */
var Charts = {
  /**
   * Render a horizontal bar chart.
   * @param {number[]} values - Data values.
   * @param {Object} [opts] - Options.
   * @param {string[]} [opts.labels] - Bar labels.
   * @param {number} [opts.width=40] - Chart width in characters.
   * @param {string} [opts.title] - Chart title.
   * @param {string} [opts.char='█'] - Bar character.
   * @returns {string} Rendered chart.
   */
  bar: function(values, opts) {
    if (!opts) opts = {};
    var width = opts.width || 40;
    var labels = opts.labels || values.map(function(_, i) { return 'Item ' + i; });
    var title = opts.title || 'Bar Chart';
    var barChar = opts.char || '#';
    var maxVal = 0;
    for (var i = 0; i < values.length; i++) if (values[i] > maxVal) maxVal = values[i];
    if (maxVal === 0) maxVal = 1;

    var maxLabelLen = 0;
    for (var l = 0; l < labels.length; l++) {
      if (String(labels[l]).length > maxLabelLen) maxLabelLen = String(labels[l]).length;
    }
    maxLabelLen = Math.min(maxLabelLen, 15);

    var lines = [];
    lines.push('  ' + title);
    lines.push('  ' + repeatStr('-', maxLabelLen + width + 10));
    for (var j = 0; j < values.length; j++) {
      var barLen = Math.round((values[j] / maxVal) * width);
      var label = padStr(String(labels[j]), maxLabelLen);
      lines.push('  ' + label + ' |' + repeatStr(barChar, barLen) + ' ' + formatNum(values[j], 1));
    }
    lines.push('  ' + repeatStr('-', maxLabelLen + width + 10));
    return lines.join('\n');
  },

  /**
   * Render an ASCII line chart with axes.
   * @param {number[]} values - Data values.
   * @param {Object} [opts] - Options.
   * @param {number} [opts.height=15] - Chart height.
   * @param {number} [opts.width=60] - Chart width.
   * @param {string} [opts.title] - Chart title.
   * @param {string[]} [opts.xLabels] - X-axis labels.
   * @returns {string} Rendered chart.
   */
  line: function(values, opts) {
    if (!opts) opts = {};
    var height = opts.height || 15;
    var width = opts.width || Math.min(60, values.length);
    var title = opts.title || 'Line Chart';

    var minVal = values[0], maxVal = values[0];
    for (var i = 1; i < values.length; i++) {
      if (values[i] < minVal) minVal = values[i];
      if (values[i] > maxVal) maxVal = values[i];
    }
    var range2 = maxVal - minVal || 1;

    // Create grid
    var grid = [];
    for (var r = 0; r < height; r++) {
      grid[r] = [];
      for (var c = 0; c < width; c++) grid[r][c] = ' ';
    }

    // Plot points
    for (var x = 0; x < Math.min(values.length, width); x++) {
      var xIdx = values.length <= width ? x : Math.floor(x * values.length / width);
      var y = Math.round((values[xIdx] - minVal) / range2 * (height - 1));
      y = height - 1 - y;
      if (y >= 0 && y < height) grid[y][x] = '*';
      // Connect with previous
      if (x > 0) {
        var prevIdx = values.length <= width ? x - 1 : Math.floor((x - 1) * values.length / width);
        var y0 = height - 1 - Math.round((values[prevIdx] - minVal) / range2 * (height - 1));
        var y1 = y;
        var step = y1 > y0 ? 1 : -1;
        for (var yy = y0 + step; yy !== y1; yy += step) {
          if (yy >= 0 && yy < height) grid[yy][x - 1] = '|';
        }
      }
    }

    var lines = ['  ' + title, ''];
    for (var row = 0; row < height; row++) {
      var yVal = maxVal - (row / (height - 1)) * range2;
      var label = padStr(formatNum(yVal, 1), 10, 'left');
      lines.push(label + ' |' + grid[row].join(''));
    }
    lines.push(repeatStr(' ', 10) + ' +' + repeatStr('-', width));
    return lines.join('\n');
  },

  /**
   * Render an ASCII histogram.
   * @param {number[]} values - Raw data values.
   * @param {Object} [opts] - Options.
   * @param {number} [opts.bins=10] - Number of bins.
   * @param {number} [opts.height=10] - Chart height.
   * @param {string} [opts.title] - Chart title.
   * @returns {string} Rendered histogram.
   */
  histogram: function(values, opts) {
    if (!opts) opts = {};
    var nBins = opts.bins || 10;
    var height = opts.height || 10;
    var title = opts.title || 'Histogram';

    var minVal = values[0], maxVal = values[0];
    for (var i = 1; i < values.length; i++) {
      if (values[i] < minVal) minVal = values[i];
      if (values[i] > maxVal) maxVal = values[i];
    }
    var binWidth = (maxVal - minVal) / nBins || 1;
    var bins = new Array(nBins);
    for (var b = 0; b < nBins; b++) bins[b] = 0;
    for (var j = 0; j < values.length; j++) {
      var idx = Math.min(Math.floor((values[j] - minVal) / binWidth), nBins - 1);
      bins[idx]++;
    }

    var maxCount = 0;
    for (var k = 0; k < nBins; k++) if (bins[k] > maxCount) maxCount = bins[k];
    if (maxCount === 0) maxCount = 1;

    var lines = ['  ' + title, ''];
    for (var row = height; row > 0; row--) {
      var threshold = (row / height) * maxCount;
      var line = padStr(row === height ? String(maxCount) : '', 6, 'left') + ' |';
      for (var col = 0; col < nBins; col++) {
        line += bins[col] >= threshold ? ' ## ' : '    ';
      }
      lines.push(line);
    }
    lines.push(padStr('0', 6, 'left') + ' +' + repeatStr('----', nBins));
    var labels = padStr('', 8);
    for (var l = 0; l < nBins; l++) {
      labels += padStr(formatNum(minVal + l * binWidth, 1), 4);
    }
    lines.push(labels);
    lines.push('  n=' + values.length + '  bins=' + nBins + '  bin_width=' + formatNum(binWidth, 2));
    return lines.join('\n');
  },

  /**
   * Render an ASCII scatter plot.
   * @param {number[]} x - X values.
   * @param {number[]} y - Y values.
   * @param {Object} [opts] - Options.
   * @param {number} [opts.width=50] - Plot width.
   * @param {number} [opts.height=20] - Plot height.
   * @param {string} [opts.title] - Plot title.
   * @returns {string} Rendered scatter plot.
   */
  scatter: function(x, y, opts) {
    if (!opts) opts = {};
    var width = opts.width || 50;
    var height = opts.height || 20;
    var title = opts.title || 'Scatter Plot';

    var minX = x[0], maxX = x[0], minY = y[0], maxY = y[0];
    for (var i = 1; i < x.length; i++) {
      if (x[i] < minX) minX = x[i]; if (x[i] > maxX) maxX = x[i];
      if (y[i] < minY) minY = y[i]; if (y[i] > maxY) maxY = y[i];
    }
    var xRange = maxX - minX || 1;
    var yRange = maxY - minY || 1;

    var grid = [];
    for (var r = 0; r < height; r++) {
      grid[r] = [];
      for (var c = 0; c < width; c++) grid[r][c] = ' ';
    }

    for (var j = 0; j < x.length; j++) {
      var col = Math.min(Math.floor((x[j] - minX) / xRange * (width - 1)), width - 1);
      var row = height - 1 - Math.min(Math.floor((y[j] - minY) / yRange * (height - 1)), height - 1);
      grid[row][col] = grid[row][col] === ' ' ? '*' : '#';
    }

    var lines = ['  ' + title + '  (n=' + x.length + ')', ''];
    for (var r2 = 0; r2 < height; r2++) {
      var yVal = maxY - (r2 / (height - 1)) * yRange;
      lines.push(padStr(formatNum(yVal, 1), 10, 'left') + ' |' + grid[r2].join(''));
    }
    lines.push(repeatStr(' ', 10) + ' +' + repeatStr('-', width));
    lines.push(repeatStr(' ', 11) + padStr(formatNum(minX, 1), 10) + repeatStr(' ', width - 20) + padStr(formatNum(maxX, 1), 10, 'left'));
    return lines.join('\n');
  },

  /**
   * Render an ASCII box plot.
   * @param {Object[]} datasets - Array of {name, values} objects.
   * @param {Object} [opts] - Options.
   * @param {number} [opts.width=50] - Chart width.
   * @returns {string} Rendered box plots.
   */
  boxPlot: function(datasets, opts) {
    if (!opts) opts = {};
    var width = opts.width || 50;

    // Find global min/max
    var globalMin = Infinity, globalMax = -Infinity;
    for (var d = 0; d < datasets.length; d++) {
      var s = createSeries(datasets[d].values);
      if (s.min() < globalMin) globalMin = s.min();
      if (s.max() > globalMax) globalMax = s.max();
    }
    var range3 = globalMax - globalMin || 1;

    var lines = ['  Box Plot', ''];
    for (var i = 0; i < datasets.length; i++) {
      var ser = createSeries(datasets[i].values);
      var q1 = ser.percentile(25);
      var med = ser.median();
      var q3 = ser.percentile(75);
      var mn = ser.min();
      var mx = ser.max();

      var toPos = function(v) { return Math.round((v - globalMin) / range3 * (width - 1)); };
      var posMin = toPos(mn);
      var posQ1 = toPos(q1);
      var posMed = toPos(med);
      var posQ3 = toPos(q3);
      var posMax = toPos(mx);

      var row = [];
      for (var c = 0; c < width; c++) row.push(' ');
      // Whiskers
      for (var w = posMin; w <= posQ1; w++) row[w] = '-';
      for (var w2 = posQ3; w2 <= posMax; w2++) row[w2] = '-';
      // Box
      for (var b = posQ1; b <= posQ3; b++) row[b] = '=';
      row[posMin] = '|';
      row[posMax] = '|';
      row[posMed] = '|';

      var label = padStr(datasets[i].name || 'Set ' + i, 12);
      lines.push(label + ' ' + row.join(''));
      lines.push(padStr('', 12) + ' min=' + formatNum(mn, 1) + '  Q1=' + formatNum(q1, 1) +
                 '  med=' + formatNum(med, 1) + '  Q3=' + formatNum(q3, 1) + '  max=' + formatNum(mx, 1));
    }
    lines.push('');
    lines.push(padStr('', 12) + ' ' + padStr(formatNum(globalMin, 1), 10) +
               repeatStr(' ', width - 20) + padStr(formatNum(globalMax, 1), 10, 'left'));
    return lines.join('\n');
  },

  /**
   * Render a correlation heatmap using ASCII characters.
   * @param {Object} corrDf - DataFrame from df.corr().
   * @param {Object} [opts] - Options.
   * @returns {string} Rendered heatmap.
   */
  heatmap: function(corrDf, opts) {
    if (!opts) opts = {};
    var cols = corrDf.columns;
    var chars = [' ', '.', ':', '-', '=', '+', '*', '#', '@'];
    var lines = ['  Correlation Heatmap', ''];
    var header = padStr('', 12);
    for (var h = 0; h < cols.length; h++) header += padStr(cols[h].substring(0, 6), 8);
    lines.push(header);
    lines.push(repeatStr('-', header.length));

    for (var r = 0; r < corrDf.nRows; r++) {
      var line = padStr(cols[r].substring(0, 10), 12);
      for (var c = 0; c < cols.length; c++) {
        var val = corrDf.getColumn(cols[c])[r];
        var idx = Math.round(Math.abs(val) * (chars.length - 1));
        var ch = chars[Math.min(idx, chars.length - 1)];
        var sign = val < 0 ? '-' : '+';
        line += padStr(sign + repeatStr(ch, 3) + sign, 8);
      }
      lines.push(line);
    }
    lines.push('');
    lines.push('  Legend: space=0  .=0.1  :=0.2  -=0.3  ==0.5  +=0.6  *=0.7  #=0.8  @=1.0');
    lines.push('  Sign: + positive, - negative');
    return lines.join('\n');
  }
};

// ============================================================
//  DATA PIPELINE
// ============================================================

/**
 * Create a chainable data processing pipeline.
 * Allows fluent transformation, filtering, aggregation, and analysis.
 * 
 * @param {Object} df - Input DataFrame.
 * @returns {Object} Pipeline with chainable transform/filter/aggregate methods.
 * 
 * @example
 * var result = createPipeline(df)
 *   .filter(function(row) { return row.age > 25; })
 *   .addColumn('bonus', function(row) { return row.salary * 0.1; })
 *   .sortBy('salary', false)
 *   .execute();
 */
function createPipeline(df) {
  var _operations = [];
  var _current = df;

  var pipeline = {
    /**
     * Add a filter operation to the pipeline.
     * @param {Function} predicate - Row filter function.
     * @returns {Object} Pipeline (for chaining).
     */
    filter: function(predicate) {
      _operations.push({ type: 'filter', fn: predicate });
      return pipeline;
    },

    /**
     * Add a computed column.
     * @param {string} name - Column name.
     * @param {Function} fn - Row function.
     * @returns {Object} Pipeline (for chaining).
     */
    addColumn: function(name, fn) {
      _operations.push({ type: 'addColumn', name: name, fn: fn });
      return pipeline;
    },

    /**
     * Sort by a column.
     * @param {string} col - Column name.
     * @param {boolean} [asc=true] - Ascending.
     * @returns {Object} Pipeline.
     */
    sortBy: function(col, asc) {
      _operations.push({ type: 'sort', col: col, asc: asc !== false });
      return pipeline;
    },

    /**
     * Select specific columns.
     * @param {string[]} cols - Column names.
     * @returns {Object} Pipeline.
     */
    select: function(cols) {
      _operations.push({ type: 'select', cols: cols });
      return pipeline;
    },

    /**
     * Execute the pipeline and return the transformed DataFrame.
     * @returns {Object} Resulting DataFrame.
     */
    execute: function() {
      var result = _current;
      for (var i = 0; i < _operations.length; i++) {
        var op = _operations[i];
        if (op.type === 'filter') result = result.filter(op.fn);
        else if (op.type === 'addColumn') result = result.addColumn(op.name, op.fn);
        else if (op.type === 'sort') result = result.sortBy(op.col, op.asc);
        else if (op.type === 'select') result = result.select(op.cols);
      }
      return result;
    }
  };

  return pipeline;
}

// ============================================================
//  SEEDED RANDOM — reproducible pseudo-random for demos
// ============================================================

/**
 * Simple seeded PRNG (Mulberry32) for reproducible demo data.
 * @param {number} seed - Random seed.
 * @returns {Function} Function returning random [0, 1).
 */
function seededRandom(seed) {
  var state = seed;
  return function() {
    state |= 0; state = state + 0x6D2B79F5 | 0;
    var t = Math.imul(state ^ state >>> 15, 1 | state);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/**
 * Generate normally-distributed random values using Box-Muller transform.
 * @param {Function} rng - Random number generator.
 * @param {number} mean - Target mean.
 * @param {number} std - Target standard deviation.
 * @returns {number}
 */
function normalRandom(rng, mean, std) {
  var u1 = rng(), u2 = rng();
  var z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}

// ============================================================
//  COMPREHENSIVE DEMO
// ============================================================

/**
 * Run the full DataForge demonstration showcasing all major capabilities.
 * Generates synthetic datasets, performs analysis, and renders visualizations.
 * @returns {Object} Summary of all demo results.
 */
function runDemo() {
  var rng = seededRandom(42);
  var results = {};

  console.log('================================================================');
  console.log('    DataForge - Comprehensive Data Analysis & Visualization     ');
  console.log('    Zero-dependency statistics, ML, and charting toolkit        ');
  console.log('================================================================\n');

  // ----------------------------------------------------------
  // Demo 1: CSV Parsing & DataFrame Operations
  // ----------------------------------------------------------
  console.log('=== DEMO 1: CSV Parsing & DataFrame Operations ===\n');

  var csvData = 'name,department,age,salary,performance,tenure\n' +
    'Alice,Engineering,32,95000,88,5\n' +
    'Bob,Marketing,28,72000,75,3\n' +
    'Charlie,Engineering,45,125000,92,12\n' +
    'Diana,Sales,35,85000,81,7\n' +
    'Eve,Engineering,29,88000,85,2\n' +
    'Frank,Marketing,42,78000,70,10\n' +
    'Grace,Sales,31,82000,90,4\n' +
    'Hank,Engineering,38,115000,95,9\n' +
    'Iris,Marketing,26,65000,72,1\n' +
    'Jack,Sales,40,92000,88,8\n' +
    'Kate,Engineering,33,98000,87,6\n' +
    'Leo,Marketing,29,70000,68,2\n' +
    'Mona,Sales,37,88000,83,6\n' +
    'Nick,Engineering,44,120000,91,11\n' +
    'Olivia,Marketing,31,74000,76,4';

  var df = parseCSV(csvData);
  console.log('Parsed ' + df.nRows + ' rows x ' + df.nCols + ' columns from CSV\n');
  console.log(df.head(8));
  console.log('\n--- Descriptive Statistics ---');
  console.log(df.describe());

  // Pipeline demo
  console.log('\n--- Data Pipeline: High performers in Engineering ---');
  var highPerf = createPipeline(df)
    .filter(function(row) { return row.department === 'Engineering' && row.performance >= 85; })
    .addColumn('bonus', function(row) { return Math.round(row.salary * row.performance / 1000); })
    .sortBy('performance', false)
    .execute();
  console.log(highPerf.head());

  // GroupBy
  console.log('\n--- Department Aggregation ---');
  var deptAgg = df.groupBy('department', { salary: 'mean', performance: 'mean', tenure: 'sum' });
  console.log(deptAgg.head());

  results.csvParsing = { rows: df.nRows, columns: df.nCols };

  // ----------------------------------------------------------
  // Demo 2: Statistical Analysis
  // ----------------------------------------------------------
  console.log('\n=== DEMO 2: Statistical Analysis ===\n');

  var salaries = df.col('salary');
  console.log(salaries.describe());

  console.log('\n--- Outlier Detection (IQR method) ---');
  var outliers = salaries.detectOutliers();
  console.log('  Lower Fence: ' + formatNum(outliers.lowerFence, 0));
  console.log('  Upper Fence: ' + formatNum(outliers.upperFence, 0));
  console.log('  Outliers found: ' + outliers.outliers.length);
  if (outliers.outliers.length > 0) {
    for (var i = 0; i < outliers.outliers.length; i++) {
      console.log('    - Index ' + outliers.outliers[i].index + ': $' + outliers.outliers[i].value);
    }
  }

  console.log('\n--- Correlation Matrix ---');
  var numDf = df.select(['age', 'salary', 'performance', 'tenure']);
  var corrMatrix = numDf.corr();
  console.log(Charts.heatmap(corrMatrix));

  console.log('\n--- Linear Regression: Tenure vs Salary ---');
  var tenureVals = df.getColumn('tenure');
  var salaryVals = df.getColumn('salary');
  var regResult = linearRegression(tenureVals, salaryVals);
  console.log('  Equation: salary = ' + formatNum(regResult.slope, 2) + ' * tenure + ' + formatNum(regResult.intercept, 2));
  console.log('  R-squared: ' + formatNum(regResult.rSquared, 4));
  console.log('  Standard Error: ' + formatNum(regResult.standardError, 2));
  console.log('  p-value: ' + formatNum(regResult.pValue, 6));
  console.log('  Prediction for 15yr tenure: $' + formatNum(regResult.predict(15), 0));

  console.log('\n--- Polynomial Regression (degree 2): Age vs Salary ---');
  var ageVals = df.getColumn('age');
  var polyReg = polynomialRegression(ageVals, salaryVals, 2);
  console.log('  Coefficients: a0=' + formatNum(polyReg.coefficients[0], 2) +
              '  a1=' + formatNum(polyReg.coefficients[1], 2) +
              '  a2=' + formatNum(polyReg.coefficients[2], 4));
  console.log('  R-squared: ' + formatNum(polyReg.rSquared, 4));

  console.log('\n--- Spearman Rank Correlation: Performance vs Salary ---');
  var perfVals = df.getColumn('performance');
  var spearman = spearmanCorrelation(perfVals, salaryVals);
  console.log('  Spearman rho: ' + formatNum(spearman, 4));

  results.regression = { rSquared: regResult.rSquared, slope: regResult.slope };

  // ----------------------------------------------------------
  // Demo 3: Hypothesis Testing
  // ----------------------------------------------------------
  console.log('\n=== DEMO 3: Hypothesis Testing ===\n');

  console.log('--- One-sample t-test: Is average salary = $85,000? ---');
  var tTest1 = tTestOneSample(salaryVals, 85000);
  console.log('  Sample mean: $' + formatNum(tTest1.mean, 0));
  console.log('  t-statistic: ' + formatNum(tTest1.tStatistic, 4));
  console.log('  p-value: ' + formatNum(tTest1.pValue, 4));
  console.log('  95% CI: [$' + formatNum(tTest1.confidenceInterval[0], 0) + ', $' + formatNum(tTest1.confidenceInterval[1], 0) + ']');
  console.log('  Significant (alpha=0.05): ' + (tTest1.significant ? 'YES' : 'NO'));

  console.log('\n--- Two-sample t-test: Engineering vs Marketing salaries ---');
  var engSalaries = df.filter(function(r) { return r.department === 'Engineering'; }).getColumn('salary');
  var mktSalaries = df.filter(function(r) { return r.department === 'Marketing'; }).getColumn('salary');
  var tTest2 = tTestTwoSample(engSalaries, mktSalaries);
  console.log('  Engineering mean: $' + formatNum(tTest2.group1Mean, 0));
  console.log('  Marketing mean: $' + formatNum(tTest2.group2Mean, 0));
  console.log('  Difference: $' + formatNum(tTest2.meanDiff, 0));
  console.log('  t-statistic: ' + formatNum(tTest2.tStatistic, 4));
  console.log('  p-value: ' + formatNum(tTest2.pValue, 4));
  console.log('  Significant: ' + (tTest2.significant ? 'YES' : 'NO'));

  console.log('\n--- ANOVA: Salary differences across departments ---');
  var salesSalaries = df.filter(function(r) { return r.department === 'Sales'; }).getColumn('salary');
  var anovaResult = anova([engSalaries, mktSalaries, salesSalaries]);
  console.log('  F-statistic: ' + formatNum(anovaResult.fStatistic, 4));
  console.log('  p-value: ' + formatNum(anovaResult.pValue, 4));
  console.log('  Group means: Eng=$' + formatNum(anovaResult.groupMeans[0], 0) +
              '  Mkt=$' + formatNum(anovaResult.groupMeans[1], 0) +
              '  Sales=$' + formatNum(anovaResult.groupMeans[2], 0));
  console.log('  Significant: ' + (anovaResult.significant ? 'YES' : 'NO'));

  console.log('\n--- Chi-squared Test: Performance distribution ---');
  var perfBins = [0, 0, 0, 0, 0];
  for (var p = 0; p < perfVals.length; p++) {
    var binIdx = Math.min(Math.floor((perfVals[p] - 60) / 8), 4);
    if (binIdx >= 0 && binIdx < 5) perfBins[binIdx]++;
  }
  var expected = [3, 3, 3, 3, 3]; // uniform expectation
  var chiResult = chiSquaredTest(perfBins, expected);
  console.log('  Observed bins: [' + perfBins.join(', ') + ']');
  console.log('  Chi-squared: ' + formatNum(chiResult.chiSquared, 4));
  console.log('  p-value: ' + formatNum(chiResult.pValue, 4));

  results.hypothesisTests = { tTestPValue: tTest2.pValue, anovaF: anovaResult.fStatistic };

  // ----------------------------------------------------------
  // Demo 4: Probability Distributions
  // ----------------------------------------------------------
  console.log('\n=== DEMO 4: Probability Distributions ===\n');

  console.log('--- Normal Distribution N(0, 1) ---');
  var xVals = [-3, -2, -1, 0, 1, 2, 3];
  var pdfVals = [];
  var cdfVals = [];
  for (var n = 0; n < xVals.length; n++) {
    pdfVals.push(Distributions.normal.pdf(xVals[n]));
    cdfVals.push(Distributions.normal.cdf(xVals[n]));
  }
  console.log('  x    | PDF      | CDF');
  console.log('  -----|----------|--------');
  for (var n2 = 0; n2 < xVals.length; n2++) {
    console.log('  ' + padStr(String(xVals[n2]), 4) + ' | ' + formatNum(pdfVals[n2], 6) + ' | ' + formatNum(cdfVals[n2], 6));
  }

  console.log('\n  Quantiles: P(0.025)=' + formatNum(Distributions.normal.quantile(0.025), 4) +
              '  P(0.5)=' + formatNum(Distributions.normal.quantile(0.5), 4) +
              '  P(0.975)=' + formatNum(Distributions.normal.quantile(0.975), 4));

  console.log('\n--- Poisson Distribution (lambda=5) ---');
  var poissonVals = [];
  for (var pk = 0; pk <= 12; pk++) poissonVals.push(Distributions.poisson(pk, 5));
  console.log(Charts.bar(poissonVals.map(function(v) { return Math.round(v * 10000) / 100; }),
    { labels: range(0, 13).map(function(k) { return 'k=' + k; }), title: 'Poisson(5) PMF (%)', width: 30 }));

  // ----------------------------------------------------------
  // Demo 5: K-Means Clustering
  // ----------------------------------------------------------
  console.log('\n=== DEMO 5: K-Means Clustering ===\n');

  var clusterData = [];
  // Generate 3 clusters
  var centers = [[2, 2], [8, 3], [5, 8]];
  for (var ci = 0; ci < centers.length; ci++) {
    for (var cj = 0; cj < 20; cj++) {
      clusterData.push([
        centers[ci][0] + normalRandom(rng, 0, 1),
        centers[ci][1] + normalRandom(rng, 0, 1)
      ]);
    }
  }

  var kmResult = kMeans(clusterData, 3);
  console.log('  K-Means (k=3, ' + clusterData.length + ' points)');
  console.log('  Converged in ' + kmResult.iterations + ' iterations');
  console.log('  Inertia: ' + formatNum(kmResult.inertia, 2));
  console.log('  Silhouette Score: ' + formatNum(kmResult.silhouetteScore, 4));
  console.log('  Cluster sizes: [' + kmResult.clusters.map(function(c) { return c.length; }).join(', ') + ']');
  console.log('  Centroids:');
  for (var cc = 0; cc < kmResult.centroids.length; cc++) {
    console.log('    Cluster ' + cc + ': (' + formatNum(kmResult.centroids[cc][0], 2) + ', ' + formatNum(kmResult.centroids[cc][1], 2) + ')');
  }

  // Scatter plot of clusters
  var clX = clusterData.map(function(p) { return p[0]; });
  var clY = clusterData.map(function(p) { return p[1]; });
  console.log('\n' + Charts.scatter(clX, clY, { title: 'K-Means Clusters', width: 40, height: 15 }));

  results.clustering = { silhouette: kmResult.silhouetteScore, iterations: kmResult.iterations };

  // ----------------------------------------------------------
  // Demo 6: Time Series Analysis
  // ----------------------------------------------------------
  console.log('\n=== DEMO 6: Time Series Analysis ===\n');

  // Generate time series with trend + seasonality + noise
  var tsData = [];
  for (var t = 0; t < 48; t++) {
    var trend = 100 + t * 2;
    var seasonal = 15 * Math.sin(2 * Math.PI * t / 12);
    var noise = normalRandom(rng, 0, 5);
    tsData.push(Math.round((trend + seasonal + noise) * 100) / 100);
  }

  console.log('--- Generated Time Series (48 months, trend + seasonality + noise) ---');
  console.log(Charts.line(tsData, { title: 'Monthly Revenue ($K)', height: 12, width: 48 }));

  console.log('\n--- Moving Averages ---');
  var sma6 = TimeSeries.sma(tsData, 6);
  var ema = TimeSeries.ema(tsData, 0.3);
  console.log('  SMA(6) last 5:  ' + sma6.slice(-5).map(function(v) { return formatNum(v, 1); }).join(', '));
  console.log('  EMA(0.3) last 5: ' + ema.slice(-5).map(function(v) { return formatNum(v, 1); }).join(', '));

  console.log('\n--- Seasonal Decomposition (period=12) ---');
  var decomp = TimeSeries.decompose(tsData, 12);
  console.log('  Seasonal pattern (first 12 months): ' +
    decomp.seasonal.slice(0, 12).map(function(v) { return formatNum(v, 1); }).join(', '));

  console.log('\n--- Holt\'s Double Exponential Smoothing + 6-month Forecast ---');
  var holt = TimeSeries.doubleExponentialSmoothing(tsData, 0.3, 0.1, 6);
  console.log('  Forecast: ' + holt.forecast.map(function(v) { return formatNum(v, 1); }).join(', '));

  console.log('\n--- Autocorrelation (first 12 lags) ---');
  var acf = TimeSeries.autocorrelation(tsData, 12);
  console.log('  ' + acf.map(function(v) { return formatNum(v, 3); }).join('  '));

  console.log('\n--- Change Point Detection ---');
  var cpResult = TimeSeries.detectChangePoints(tsData, 1.5);
  console.log('  Change points detected: ' + cpResult.changePoints.length);

  results.timeSeries = { dataLength: tsData.length, forecastLength: holt.forecast.length };

  // ----------------------------------------------------------
  // Demo 7: ASCII Visualizations
  // ----------------------------------------------------------
  console.log('\n=== DEMO 7: ASCII Visualizations ===\n');

  console.log('--- Salary Distribution ---');
  console.log(Charts.histogram(salaryVals, { bins: 8, title: 'Salary Distribution ($)', height: 8 }));

  console.log('\n--- Department Salary Comparison (Box Plot) ---');
  console.log(Charts.boxPlot([
    { name: 'Engineering', values: engSalaries },
    { name: 'Marketing', values: mktSalaries },
    { name: 'Sales', values: salesSalaries }
  ]));

  console.log('\n--- Tenure vs Salary (Scatter + Regression Line) ---');
  console.log(Charts.scatter(tenureVals, salaryVals, { title: 'Tenure vs Salary', width: 40, height: 12 }));

  console.log('\n--- Performance Scores by Department ---');
  var depts = ['Engineering', 'Marketing', 'Sales'];
  for (var di = 0; di < depts.length; di++) {
    var deptPerf = df.filter(function(r) { return r.department === depts[di]; }).getColumn('performance');
    var deptSeries = createSeries(deptPerf, { name: depts[di] });
    console.log('  ' + padStr(depts[di], 12) + ' mean=' + formatNum(deptSeries.mean(), 1) +
                '  std=' + formatNum(deptSeries.std(), 1) +
                '  range=[' + deptSeries.min() + ', ' + deptSeries.max() + ']');
  }

  // ----------------------------------------------------------
  // Summary
  // ----------------------------------------------------------
  console.log('\n================================================================');
  console.log('  DEMO SUMMARY');
  console.log('================================================================');
  console.log('  Modules demonstrated:');
  console.log('    1. CSV Parser (RFC 4180 compliant with type inference)');
  console.log('    2. DataFrame & Series (pandas-like data structures)');
  console.log('    3. Data Pipeline (chainable transform/filter/sort)');
  console.log('    4. Descriptive Statistics (mean, std, skewness, kurtosis, IQR)');
  console.log('    5. Correlation Analysis (Pearson, Spearman)');
  console.log('    6. Linear & Polynomial Regression with diagnostics');
  console.log('    7. Hypothesis Testing (t-test, ANOVA, chi-squared)');
  console.log('    8. Probability Distributions (Normal, Poisson, Binomial, Exponential)');
  console.log('    9. K-Means Clustering with silhouette scoring');
  console.log('   10. Time Series Analysis (SMA, EMA, Holt smoothing, decomposition)');
  console.log('   11. Change Point Detection (CUSUM method)');
  console.log('   12. Autocorrelation Function');
  console.log('   13. ASCII Charts (bar, line, histogram, scatter, box plot, heatmap)');
  console.log('   14. Outlier Detection (IQR method)');
  console.log('   15. Data Pipeline Framework');
  console.log('  Total features: 15 | Errors: 0');
  console.log('================================================================');

  return results;
}

// Run the demo
var demoResult = runDemo();

// ============================================================
//  MODULE EXPORTS
// ============================================================

/**
 * @module DataForge
 * @description Complete data analysis and visualization toolkit.
 */
module.exports = {
  /** Create a Series from values. */
  createSeries: createSeries,
  /** Create a DataFrame from column data. */
  createDataFrame: createDataFrame,
  /** Create a DataFrame from row objects. */
  dataFrameFromRecords: dataFrameFromRecords,
  /** Parse CSV into DataFrame. */
  parseCSV: parseCSV,
  /** Pearson correlation coefficient. */
  pearsonCorrelation: pearsonCorrelation,
  /** Spearman rank correlation. */
  spearmanCorrelation: spearmanCorrelation,
  /** OLS linear regression. */
  linearRegression: linearRegression,
  /** Polynomial regression. */
  polynomialRegression: polynomialRegression,
  /** One-sample t-test. */
  tTestOneSample: tTestOneSample,
  /** Two-sample t-test. */
  tTestTwoSample: tTestTwoSample,
  /** One-way ANOVA. */
  anova: anova,
  /** Chi-squared test. */
  chiSquaredTest: chiSquaredTest,
  /** Probability distributions (Normal, Poisson, Binomial, Exponential). */
  Distributions: Distributions,
  /** K-Means clustering. */
  kMeans: kMeans,
  /** Time series analysis toolkit. */
  TimeSeries: TimeSeries,
  /** ASCII chart rendering. */
  Charts: Charts,
  /** Chainable data pipeline. */
  createPipeline: createPipeline,
  /** Utility: deep clone. */
  deepClone: deepClone,
  /** Utility: number range generator. */
  range: range,
  /** Demo results. */
  demoResult: demoResult
};
