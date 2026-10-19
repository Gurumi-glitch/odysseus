import test from 'node:test';
import assert from 'node:assert/strict';
import {
  TEMPERATURE_DEFAULT_STOP, TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_STEP,
  temperatureToSlider, sliderToTemperature, formatTemperature,
} from '../static/js/generationSettings.js';

test('the Default stop is the left end, one step below 0, so 0..2 reads left to right', () => {
  assert.equal(TEMPERATURE_MIN, 0);
  assert.equal(TEMPERATURE_MAX, 2);
  assert.equal(TEMPERATURE_STEP, 0.1);
  assert.equal(TEMPERATURE_DEFAULT_STOP, -0.1);
  const labels = ['-0.1', '0', '0.1', '1', '2'].map(formatTemperature);
  assert.deepEqual(labels, ['Default', '0.0', '0.1', '1.0', '2.0']);
});

test('unset temperature sits on the Default stop and is labelled Default', () => {
  for (const unset of [null, undefined]) {
    assert.equal(temperatureToSlider(unset), TEMPERATURE_DEFAULT_STOP);
    assert.equal(formatTemperature(temperatureToSlider(unset)), 'Default');
  }
});

test('the Default stop stores null, every other stop stores its number', () => {
  assert.equal(sliderToTemperature('-0.1'), null);
  assert.equal(sliderToTemperature('0'), 0);
  assert.equal(sliderToTemperature('1.3'), 1.3);
  assert.equal(sliderToTemperature('2'), 2);
});

test('a set value round-trips and returning to Default works', () => {
  assert.equal(sliderToTemperature(temperatureToSlider(0.7)), 0.7);
  assert.equal(sliderToTemperature(temperatureToSlider(0)), 0);
  assert.equal(formatTemperature(0.7), '0.7');
  assert.equal(formatTemperature(2), '2.0');
  assert.equal(sliderToTemperature(temperatureToSlider(sliderToTemperature('-0.1'))), null);
});
