// AI Backend Wrappers — Claude (Anthropic) + GPT (OpenAI)

const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';
const OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions';

export async function callClaude(systemPrompt, messages, apiKey, model = 'claude-sonnet-4-20250514') {
  const formattedMessages = messages.map(m => ({
    role: m.role === 'user' ? 'user' : 'assistant',
    content: m.content,
  }));

  const response = await fetch(ANTHROPIC_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model,
      max_tokens: 4096,
      system: systemPrompt,
      messages: formattedMessages,
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Claude API error (${response.status}): ${err}`);
  }

  const data = await response.json();
  return data.content[0].text;
}

export async function callGPT(systemPrompt, messages, apiKey, model = 'gpt-4o') {
  const formattedMessages = [
    { role: 'system', content: systemPrompt },
    ...messages.map(m => ({
      role: m.role === 'user' ? 'user' : 'assistant',
      content: m.content,
    })),
  ];

  const response = await fetch(OPENAI_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      max_tokens: 4096,
      messages: formattedMessages,
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`GPT API error (${response.status}): ${err}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

export async function callAI(systemPrompt, messages, provider, apiKey, model) {
  if (provider === 'anthropic') {
    return callClaude(systemPrompt, messages, apiKey, model);
  } else if (provider === 'openai') {
    return callGPT(systemPrompt, messages, apiKey, model);
  }
  throw new Error(`Unknown provider: ${provider}`);
}
