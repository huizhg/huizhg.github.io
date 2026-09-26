// Calm Shiki themes built from the site palette. Every token color keeps
// at least 4.5:1 contrast against the --card background it sits on.

const theme = (name, type, c) => ({
  name,
  type,
  colors: { 'editor.background': c.bg, 'editor.foreground': c.fg },
  tokenColors: [
    { settings: { foreground: c.fg } },
    { scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: c.comment, fontStyle: 'italic' } },
    { scope: ['keyword', 'storage', 'storage.type', 'storage.modifier', 'keyword.operator.logical', 'keyword.operator.new'], settings: { foreground: c.keyword } },
    { scope: ['string', 'string.quoted', 'punctuation.definition.string'], settings: { foreground: c.string } },
    { scope: ['constant.numeric', 'constant.language', 'constant.character', 'support.constant'], settings: { foreground: c.number } },
    { scope: ['entity.name.function', 'support.function', 'meta.function-call.generic'], settings: { foreground: c.func } },
    { scope: ['entity.name.type', 'entity.name.class', 'support.class', 'support.type'], settings: { foreground: c.type } },
    { scope: ['variable.parameter', 'variable.language'], settings: { foreground: c.param } },
    { scope: ['punctuation', 'keyword.operator'], settings: { foreground: c.punct } },
  ],
});

export const codeLight = theme('kz-light', 'light', {
  bg: '#F2EEE5',
  fg: '#1C1B19',
  comment: '#5E5A52',
  keyword: '#1F5C5A',
  string: '#7A4A1F',
  number: '#8C3A2B',
  func: '#2F4E78',
  type: '#5B4478',
  param: '#45423C',
  punct: '#45423C',
});

export const codeDark = theme('kz-dark', 'dark', {
  bg: '#1F1D1A',
  fg: '#ECE8DF',
  comment: '#A39E93',
  keyword: '#6FB3AE',
  string: '#D9A066',
  number: '#E3A08C',
  func: '#9DB8DE',
  type: '#C3A8E0',
  param: '#C9C4B9',
  punct: '#C9C4B9',
});
