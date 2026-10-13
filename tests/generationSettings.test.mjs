import test from 'node:test';
import assert from 'node:assert/strict';
import { TEMPERATURE_DEFAULT_STOP, temperatureToSlider, sliderToTemperature, formatTemperature } from '../static/js/generationSettings.js';

test('unset temperature sits on the Default stop and is labelled Default', () => {
  for (const unset of [null, undefined]) {
    assert.equal(temperatureToSlider(unset), TEMPERATURE_DEFAULT_STOP);
    assert.equal(formatTemperature(temperatureToSlider(unset)), 'Default');
  }
});

test('the Default stop stores null, every other stop stores its number', () => {
  assert.equal(sliderToTemperature(String(TEMPERATURE_DEFAULT_STOP)), null);
  assert.equal(sliderToTemperature('0'), 0);
  assert.equal(sliderToTemperature('1.3'), 1.3);
  assert.equal(sliderToTemperature('2'), 2);
});

test('a set value round-trips and returning to Default works', () => {
  assert.equal(sliderToTemperature(temperatureToSlider(0.7)), 0.7);
  assert.equal(formatTemperature(0.7), '0.7');
  assert.equal(formatTemperature(2), '2.0');
  assert.equal(sliderToTemperature(temperatureToSlider(sliderToTemperature('2.1'))), null);
});
