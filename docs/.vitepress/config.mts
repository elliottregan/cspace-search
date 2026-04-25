import { defineConfig } from 'vitepress';

export default defineConfig({
  title: 'cspace-search',
  description:
    'Local-first semantic search for commits, code, and project context. Single binary, no daemon.',

  // Pretty URLs without `.html`. Hosting providers that serve directories
  // as `index.html` (Netlify, GitHub Pages, etc.) handle this transparently.
  cleanUrls: true,

  lastUpdated: true,

  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/quickstart' },
      { text: 'Architecture', link: '/architecture/what-happens-during-a-search' },
      {
        text: 'GitHub',
        link: 'https://github.com/elliottregan/cspace-search',
      },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Getting started',
          items: [
            { text: 'Quick start', link: '/guide/quickstart' },
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Configuration', link: '/guide/configuration' },
          ],
        },
      ],
      '/architecture/': [
        {
          text: 'Architecture',
          items: [
            {
              text: 'What actually happens during a search',
              link: '/architecture/what-happens-during-a-search',
            },
          ],
        },
      ],
    },

    socialLinks: [
      {
        icon: 'github',
        link: 'https://github.com/elliottregan/cspace-search',
      },
    ],

    editLink: {
      pattern:
        'https://github.com/elliottregan/cspace-search/edit/main/docs/:path',
      text: 'Edit this page on GitHub',
    },

    footer: {
      message: 'Released under the PolyForm Perimeter 1.0.0 license.',
      copyright: 'Copyright © Elliott Regan',
    },
  },
});
