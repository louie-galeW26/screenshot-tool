import { anthropic } from "@ai-sdk/anthropic";
import { generateObject } from "ai";
import Browserbase from "@browserbasehq/sdk";
import { chromium, type Page, type Browser } from "playwright-core";
import sharp from "sharp";
import { z } from "zod";
import type {
  ParsedQuery,
  ContentFinding,
  BrowseResult,
  ProgressCallback,
} from "./types";

const TOTAL_TIMEOUT_MS = 50_000; // Stay under Vercel Hobby 60s function cap
const MAX_FINDINGS = 20;
const MAX_TEXT_LENGTH = 4000;
const MAX_SCREENSHOT_HEIGHT = 1000;
const MAX_VISION_DIMENSION = 7500; // Claude's hard limit is 8000px, keep margin

// --- Zod schemas for structured AI output ---

const pageAnalysisSchema = z.object({
  relevantSections: z.array(
    z.object({
      description: z.string(),
      locatorStrategy: z.enum(["text", "heading", "css-selector"]),
      locatorValue: z.string(),
      extractedText: z.string(),
      relevanceScore: z.number().min(0).max(1),
    })
  ),
  linksToFollow: z.array(
    z.object({
      url: z.string(),
      reason: z.string(),
      priority: z.number().min(1).max(3),
    })
  ),
  pageSummary: z.string(),
});

type PageAnalysis = z.infer<typeof pageAnalysisSchema>;
type RelevantSection = PageAnalysis["relevantSections"][number];

// --- Browser session management ---

async function createBrowserSession(): Promise<{
  browser: Browser;
  page: Page;
  sessionId: string;
}> {
  const bb = new Browserbase({
    apiKey: process.env.BROWSERBASE_API_KEY!,
  });

  const session = await bb.sessions.create({
    projectId: process.env.BROWSERBASE_PROJECT_ID!,
  });

  const browser = await chromium.connectOverCDP(session.connectUrl);
  const context = browser.contexts()[0];
  const page = context.pages()[0];

  return { browser, page, sessionId: session.id };
}

// --- Overlay dismissal ---

async function dismissOverlays(page: Page): Promise<void> {
  const dismissSelectors = [
    'button:has-text("Accept")',
    'button:has-text("Accept All")',
    'button:has-text("Got it")',
    'button:has-text("I agree")',
    'button:has-text("OK")',
    'button:has-text("Close")',
    '[aria-label*="close" i]',
    '[aria-label*="dismiss" i]',
    '[aria-label*="accept" i]',
    ".cookie-banner button",
    "#cookie-banner button",
  ];

  for (const selector of dismissSelectors) {
    try {
      const el = page.locator(selector).first();
      if (await el.isVisible({ timeout: 500 })) {
        await el.click({ timeout: 1000 });
        await page.waitForTimeout(300);
      }
    } catch {
      // Best-effort, continue
    }
  }
}

// --- Image resizing for Claude vision API ---

async function resizeForVision(buffer: Buffer): Promise<Buffer> {
  const MAX_BYTES = 4.5 * 1024 * 1024; // Claude limit is 5MB, keep margin
  const image = sharp(buffer);
  const metadata = await image.metadata();
  const width = metadata.width ?? 0;
  const height = metadata.height ?? 0;

  // Resize dimensions if needed
  let pipeline = image;
  if (width > MAX_VISION_DIMENSION || height > MAX_VISION_DIMENSION) {
    const scale = MAX_VISION_DIMENSION / Math.max(width, height);
    pipeline = pipeline.resize({
      width: Math.floor(width * scale),
      height: Math.floor(height * scale),
      fit: "inside",
    });
  }

  // Convert to JPEG at progressively lower quality until under size limit
  for (const quality of [90, 80, 70, 60, 50]) {
    const out = await pipeline.jpeg({ quality }).toBuffer();
    if (out.length <= MAX_BYTES) return out;
  }

  // Last resort: aggressive downscale + low quality
  return pipeline
    .resize({ width: 2000, fit: "inside" })
    .jpeg({ quality: 60 })
    .toBuffer();
}

// --- Page analysis with Claude Vision ---

async function analyzePageForContent(
  page: Page,
  screenshotBuffer: Buffer,
  query: ParsedQuery
): Promise<PageAnalysis> {
  // Resize if needed to fit Claude's vision limits
  const visionBuffer = await resizeForVision(screenshotBuffer);

  // Extract page text
  const innerText = await page.evaluate(() =>
    document.body.innerText.slice(0, 4000)
  );

  // Extract links
  const links = await page.evaluate(() => {
    const anchors = Array.from(document.querySelectorAll("a[href]"));
    return anchors
      .map((a) => ({
        text: (a as HTMLAnchorElement).innerText.trim().slice(0, 80),
        href: (a as HTMLAnchorElement).href,
      }))
      .filter((l) => l.text && l.href.startsWith("http"))
      .slice(0, 30);
  });

  const linksText = links
    .map((l) => `- "${l.text}" → ${l.href}`)
    .join("\n");

  const result = await generateObject({
    model: anthropic("claude-sonnet-4-20250514"),
    schema: pageAnalysisSchema,
    messages: [
      {
        role: "system",
        content: `You are analyzing a webpage screenshot to find specific content for a user.

You will receive a screenshot of the page, its text content, and a list of links.

Your job:
- Identify sections that contain content matching the search objective
- For each relevant section, provide a way to locate it in the DOM
- Suggest links worth following to find more relevant content
- Be selective, not exhaustive. Only mark sections as truly relevant.
- Score relevance honestly (0-1).

For locatorStrategy, prefer in order:
1. "text" — a unique 10-40 char substring visible in/near the section
2. "heading" — the text of a heading (h1-h6) above or in the section
3. "css-selector" — only if text-based approaches won't work

For linksToFollow: only suggest same-domain links likely to have more relevant content. Ignore social media, login, privacy policy, and footer links. Priority 3 = most likely relevant.`,
      },
      {
        role: "user",
        content: [
          {
            type: "image",
            image: visionBuffer,
          },
          {
            type: "text",
            text: `Page URL: ${page.url()}
Page text (first ${MAX_TEXT_LENGTH} chars):
${innerText}

Links on this page:
${linksText}

Search objective: "${query.searchObjective}"

Analyze this page and identify relevant content sections and links to follow.`,
          },
        ],
      },
    ],
  });

  return result.object;
}

// --- Element screenshot capture ---

async function captureElementScreenshot(
  page: Page,
  section: RelevantSection
): Promise<{ screenshotBuffer: Buffer; extractedText: string }> {
  let element;

  // Fallback chain for locating elements
  if (section.locatorStrategy === "text") {
    element = page.getByText(section.locatorValue, { exact: false }).first();
  } else if (section.locatorStrategy === "heading") {
    element = page
      .getByRole("heading", { name: section.locatorValue })
      .first();
  } else {
    element = page.locator(section.locatorValue).first();
  }

  // Try to find the element with a timeout
  await element.waitFor({ state: "visible", timeout: 3000 });

  // Walk up to a meaningful container
  const containerSelector = await element.evaluate((el: Element) => {
    let current = el.parentElement;
    const containerTags = ["SECTION", "ARTICLE", "MAIN"];
    let bestContainer: Element = el;

    for (let i = 0; i < 10 && current; i++) {
      if (containerTags.includes(current.tagName)) {
        bestContainer = current;
        break;
      }
      if (
        current.tagName === "DIV" &&
        (current as HTMLElement).offsetHeight >= 50 &&
        (current as HTMLElement).offsetHeight <= 1200
      ) {
        bestContainer = current;
      }
      current = current.parentElement;
    }

    // Generate a unique selector for the container
    if (bestContainer.id) return `#${bestContainer.id}`;
    const tag = bestContainer.tagName.toLowerCase();
    const parent = bestContainer.parentElement;
    if (!parent) return tag;
    const siblings = Array.from(parent.children).filter(
      (c) => c.tagName === bestContainer.tagName
    );
    const index = siblings.indexOf(bestContainer);
    return `${tag}:nth-of-type(${index + 1})`;
  });

  // Locate the container via the generated selector, falling back to the original element
  let containerLocator;
  try {
    containerLocator = page.locator(containerSelector).first();
    await containerLocator.waitFor({ state: "visible", timeout: 2000 });
  } catch {
    containerLocator = element;
  }

  // Cap screenshot height
  const box = await containerLocator.boundingBox();
  if (box && box.height > MAX_SCREENSHOT_HEIGHT) {
    const screenshotBuffer = Buffer.from(
      await page.screenshot({
        clip: {
          x: box.x,
          y: box.y,
          width: box.width,
          height: MAX_SCREENSHOT_HEIGHT,
        },
        type: "png",
      })
    );
    const extractedText = await containerLocator.innerText();
    return { screenshotBuffer, extractedText };
  }

  const screenshotBuffer = Buffer.from(
    await containerLocator.screenshot({ type: "png" })
  );
  const extractedText = await containerLocator.innerText();

  return { screenshotBuffer, extractedText };
}

// --- Domain matching ---

function isSameDomain(url: string, targetUrl: string): boolean {
  try {
    const a = new URL(url);
    const b = new URL(targetUrl);
    return (
      a.hostname === b.hostname || a.hostname.endsWith("." + b.hostname)
    );
  } catch {
    return false;
  }
}

// --- Main browsing loop ---

export async function browseForContent(
  query: ParsedQuery,
  onProgress: ProgressCallback
): Promise<BrowseResult> {
  const startTime = Date.now();
  const findings: ContentFinding[] = [];
  const pagesVisited: string[] = [];
  const errors: string[] = [];

  if (!query.targetUrl) {
    return {
      query,
      findings: [],
      pagesVisited: [],
      errors: ["No target URL provided"],
      durationMs: Date.now() - startTime,
    };
  }

  const { browser, page, sessionId } = await createBrowserSession();
  await onProgress(`Browser session created (${sessionId})`);

  try {
    const toVisit: Array<{ url: string; priority: number }> = [
      { url: query.targetUrl, priority: 3 },
    ];
    const visited = new Set<string>();

    while (
      toVisit.length > 0 &&
      visited.size < query.maxPages &&
      findings.length < MAX_FINDINGS &&
      Date.now() - startTime < TOTAL_TIMEOUT_MS
    ) {
      // Sort by priority descending, take the highest
      toVisit.sort((a, b) => b.priority - a.priority);
      const next = toVisit.shift()!;

      if (visited.has(next.url)) continue;
      visited.add(next.url);

      // Navigate
      await onProgress(`Navigating to ${next.url}...`);
      try {
        await page.goto(next.url, {
          waitUntil: "domcontentloaded",
          timeout: 30_000,
        });
        await page.waitForTimeout(2000); // Let JS render
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        errors.push(`Failed to load ${next.url}: ${msg}`);
        continue;
      }

      pagesVisited.push(next.url);

      // Check for blank/broken page
      const textLength = await page.evaluate(
        () => document.body.innerText.length
      );
      if (textLength < 50) {
        errors.push(`Page appears empty: ${next.url}`);
        continue;
      }

      // Dismiss overlays
      await dismissOverlays(page);

      // Take full-page screenshot for analysis
      const fullScreenshot = Buffer.from(
        await page.screenshot({ fullPage: true, type: "png" })
      );

      // Analyze with Claude Vision
      await onProgress(`Analyzing page content...`);
      let analysis: PageAnalysis;
      try {
        analysis = await analyzePageForContent(page, fullScreenshot, query);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        errors.push(`Analysis failed for ${next.url}: ${msg}`);
        continue;
      }

      await onProgress(
        `Found ${analysis.relevantSections.length} relevant sections on this page`
      );

      // Capture each relevant section. If element capture fails, keep the finding
      // with a viewport-region fallback so we don't lose Claude's extracted text.
      const pageTitle = await page.title();
      for (const section of analysis.relevantSections) {
        if (findings.length >= MAX_FINDINGS) break;

        let screenshotBuffer: Buffer;
        let extractedText = section.extractedText;

        try {
          const result = await captureElementScreenshot(page, section);
          screenshotBuffer = result.screenshotBuffer;
          extractedText = result.extractedText || section.extractedText;
        } catch (e) {
          // Fallback: viewport screenshot at the current scroll position.
          // Scroll to the text first if possible.
          const msg = e instanceof Error ? e.message : String(e);
          errors.push(
            `Element locator failed for "${section.description}" on ${next.url} (${msg}); using fallback screenshot`
          );

          try {
            // Try to scroll the approximate text into view
            await page.evaluate((text) => {
              const all = Array.from(document.body.querySelectorAll("*"));
              const match = all.find(
                (el) =>
                  el.childNodes.length > 0 &&
                  (el as HTMLElement).innerText?.includes(text.slice(0, 30))
              );
              if (match) {
                match.scrollIntoView({ block: "center", behavior: "instant" as ScrollBehavior });
              }
            }, section.locatorValue);
            await page.waitForTimeout(300);
          } catch {}

          screenshotBuffer = Buffer.from(
            await page.screenshot({ fullPage: false, type: "png" })
          );
        }

        findings.push({
          pageUrl: next.url,
          pageTitle,
          sectionHeading: section.description,
          extractedText,
          screenshotBuffer,
          relevanceScore: section.relevanceScore,
        });
      }

      // Queue new links (same-domain only)
      for (const link of analysis.linksToFollow) {
        if (
          !visited.has(link.url) &&
          isSameDomain(link.url, query.targetUrl)
        ) {
          toVisit.push({ url: link.url, priority: link.priority });
        }
      }
    }

    // Check for timeout
    if (Date.now() - startTime >= TOTAL_TIMEOUT_MS && findings.length > 0) {
      errors.push("Timed out — returning partial results");
    }
  } finally {
    await browser.close();
  }

  // Sort findings by relevance
  findings.sort((a, b) => b.relevanceScore - a.relevanceScore);

  return {
    query,
    findings,
    pagesVisited,
    errors,
    durationMs: Date.now() - startTime,
  };
}
