// Parse self-modification commands from AI responses
// Commands: [MEMORY:], [DIRECTIVE:], [SKILL:], [STATE:], [EVOLVE:], [CRYSTAL:]

const COMMAND_PATTERNS = [
  {
    type: 'MEMORY',
    regex: /\[MEMORY:\s*(.+?)\]/gs,
    parse: (match) => ({ content: match[1].trim() }),
  },
  {
    type: 'DIRECTIVE',
    regex: /\[DIRECTIVE:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\]/gs,
    parse: (match) => ({
      directiveType: match[1].trim(),
      scope: match[2].trim(),
      content: match[3].trim(),
      evidence: match[4].trim(),
    }),
  },
  {
    type: 'SKILL',
    regex: /\[SKILL:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\]/gs,
    parse: (match) => ({
      name: match[1].trim(),
      trigger: match[2].trim(),
      content: match[3].trim(),
    }),
  },
  {
    type: 'STATE',
    regex: /\[STATE:\s*(.+?)\s*\|\s*(.+?)\s*(?:\|\s*(\d+))?\]/gs,
    parse: (match) => ({
      key: match[1].trim(),
      value: match[2].trim(),
      ttlHours: match[3] ? parseInt(match[3].trim()) : 24,
    }),
  },
  {
    type: 'EVOLVE',
    regex: /\[EVOLVE:\s*(.+?)\s*\|\s*(.+?)\]/gs,
    parse: (match) => ({
      description: match[1].trim(),
      evidence: match[2].trim(),
    }),
  },
  {
    type: 'CRYSTAL',
    regex: /\[CRYSTAL:\s*(.+?)\]/gs,
    parse: (match) => ({ summary: match[1].trim() }),
  },
];

export function parseSelfModCommands(text) {
  const commands = [];

  for (const pattern of COMMAND_PATTERNS) {
    let match;
    // Reset regex state
    pattern.regex.lastIndex = 0;
    while ((match = pattern.regex.exec(text)) !== null) {
      commands.push({
        type: pattern.type,
        raw: match[0],
        data: pattern.parse(match),
      });
    }
  }

  return commands;
}

export function stripCommands(text) {
  let cleaned = text;
  for (const pattern of COMMAND_PATTERNS) {
    pattern.regex.lastIndex = 0;
    cleaned = cleaned.replace(pattern.regex, '');
  }
  return cleaned.trim();
}
