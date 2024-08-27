import { StreamingCallback, GenkitError } from '@genkit-ai/core';
import { z } from 'zod';
import { DocumentData } from './document.js';
import { MessageData, Part, ToolResponsePart, ToolRequestPart, CandidateData, GenerationUsage, GenerateRequest, GenerateResponseData, GenerateResponseChunkData, ModelArgument, GenerationCommonConfigSchema } from './model.js';
import { ToolArgument } from './tool.js';

/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Message represents a single role's contribution to a generation. Each message
 * can contain multiple parts (for example text and an image), and each generation
 * can contain multiple messages.
 */
declare class Message<T = unknown> implements MessageData {
    role: MessageData['role'];
    content: Part[];
    constructor(message: MessageData);
    /**
     * If a message contains a `data` part, it is returned. Otherwise, the `output()`
     * method extracts the first valid JSON object or array from the text contained in
     * the message and returns it.
     *
     * @returns The structured output contained in the message.
     */
    output(): T;
    toolResponseParts(): ToolResponsePart[];
    /**
     * Concatenates all `text` parts present in the message with no delimiter.
     * @returns A string of all concatenated text parts.
     */
    text(): string;
    /**
     * Returns the first media part detected in the message. Useful for extracting
     * (for example) an image from a generation expected to create one.
     * @returns The first detected `media` part in the message.
     */
    media(): {
        url: string;
        contentType?: string;
    } | null;
    /**
     * Returns the first detected `data` part of a message.
     * @returns The first `data` part detected in the message (if any).
     */
    data(): T | null;
    /**
     * Returns all tool request found in this message.
     * @returns Array of all tool request found in this message.
     */
    toolRequests(): ToolRequestPart[];
    /**
     * Converts the Message to a plain JS object.
     * @returns Plain JS object representing the data contained in the message.
     */
    toJSON(): MessageData;
}
/**
 * Candidate represents one of several possible generated responses from a generation
 * request. A candidate contains a single generated message along with additional
 * metadata about its generation. A generation request can create multiple candidates.
 */
declare class Candidate<O = unknown> implements CandidateData {
    /** The message this candidate generated. */
    message: Message<O>;
    /** The positional index of this candidate in the generation response. */
    index: number;
    /** Usage information about this candidate. */
    usage: GenerationUsage;
    /** The reason generation stopped for this candidate. */
    finishReason: CandidateData['finishReason'];
    /** Additional information about why the candidate stopped generating, if any. */
    finishMessage?: string;
    /** Additional provider-specific information about this candidate. */
    custom: unknown;
    /** The request that led to the generation of this candidate. */
    request?: GenerateRequest;
    constructor(candidate: CandidateData, request?: GenerateRequest);
    /**
     * If a candidate's message contains a `data` part, it is returned. Otherwise, the `output()`
     * method extracts the first valid JSON object or array from the text contained in
     * the candidate's message and returns it.
     *
     * @returns The structured output contained in the candidate.
     */
    output(): O | null;
    /**
     * Concatenates all `text` parts present in the candidate's message with no delimiter.
     * @returns A string of all concatenated text parts.
     */
    text(): string;
    /**
     * Returns the first detected media part in the candidate's message. Useful for extracting
     * (for example) an image from a generation expected to create one.
     * @returns The first detected `media` part in the candidate.
     */
    media(): {
        url: string;
        contentType?: string;
    } | null;
    /**
     * Returns the first detected `data` part of a candidate's message.
     * @returns The first `data` part detected in the candidate (if any).
     */
    data(): O | null;
    /**
     * Returns all tool request found in this candidate.
     * @returns Array of all tool request found in this candidate.
     */
    toolRequests(): ToolRequestPart[];
    /**
     * Determine whether this candidate has output that conforms to a provided schema.
     *
     * @param request A request containing output schema to validate against. If not provided, uses request embedded in candidate.
     * @returns True if output matches request schema or if no request schema is provided.
     */
    hasValidOutput(request?: GenerateRequest): boolean;
    /**
     * Appends the message generated by this candidate to the messages already
     * present in the generation request. The result of this method can be safely
     * serialized to JSON for persistence in a database.
     * @returns A serializable list of messages compatible with `generate({history})`.
     */
    toHistory(): MessageData[];
    /**
     * Converts the Candidate to a plain JS object.
     * @returns Plain JS object representing the data contained in the candidate.
     */
    toJSON(): CandidateData;
}
/**
 * GenerateResponse is the result from a `generate()` call and contains one or
 * more generated candidate messages.
 */
declare class GenerateResponse<O = unknown> implements GenerateResponseData {
    /** The potential generated messages. */
    candidates: Candidate<O>[];
    /** Usage information. */
    usage: GenerationUsage;
    /** Provider-specific response data. */
    custom: unknown;
    /** The request that generated this response. */
    request?: GenerateRequest;
    /**
     * If the selected candidate's message contains a `data` part, it is returned. Otherwise,
     * the `output()` method extracts the first valid JSON object or array from the text
     * contained in the selected candidate's message and returns it.
     *
     * @param index The candidate index from which to extract output. If not provided, finds first candidate that conforms to output schema.
     * @returns The structured output contained in the selected candidate.
     */
    output(index?: number): O | null;
    /**
     * Concatenates all `text` parts present in the candidate's message with no delimiter.
     * @param index The candidate index from which to extract text, defaults to first candidate.
     * @returns A string of all concatenated text parts.
     */
    text(index?: number): string;
    /**
     * Returns the first detected media part in the selected candidate's message. Useful for
     * extracting (for example) an image from a generation expected to create one.
     * @param index The candidate index from which to extract media, defaults to first candidate.
     * @returns The first detected `media` part in the candidate.
     */
    media(index?: number): {
        url: string;
        contentType?: string;
    } | null;
    /**
     * Returns the first detected `data` part of the selected candidate's message.
     * @param index The candidate index from which to extract data, defaults to first candidate.
     * @returns The first `data` part detected in the candidate (if any).
     */
    data(index?: number): O | null;
    /**
     * Returns all tool request found in the candidate.
     * @param index The candidate index from which to extract tool requests, defaults to first candidate.
     * @returns Array of all tool request found in the candidate.
     */
    toolRequests(index?: number): ToolRequestPart[];
    /**
     * Appends the message generated by the selected candidate to the messages already
     * present in the generation request. The result of this method can be safely
     * serialized to JSON for persistence in a database.
     * @param index The candidate index to utilize during conversion, defaults to first candidate.
     * @returns A serializable list of messages compatible with `generate({history})`.
     */
    toHistory(index?: number): MessageData[];
    constructor(response: GenerateResponseData, request?: GenerateRequest);
    toJSON(): GenerateResponseData;
}
declare class GenerateResponseChunk<T = unknown> implements GenerateResponseChunkData {
    /** The index of the candidate this chunk corresponds to. */
    index: number;
    /** The content generated in this chunk. */
    content: Part[];
    /** Custom model-specific data for this chunk. */
    custom?: unknown;
    /** Accumulated chunks for partial output extraction. */
    accumulatedChunks?: GenerateResponseChunkData[];
    constructor(data: GenerateResponseChunkData, accumulatedChunks?: GenerateResponseChunkData[]);
    /**
     * Concatenates all `text` parts present in the chunk with no delimiter.
     * @returns A string of all concatenated text parts.
     */
    text(): string;
    /**
     * Returns the first media part detected in the chunk. Useful for extracting
     * (for example) an image from a generation expected to create one.
     * @returns The first detected `media` part in the chunk.
     */
    media(): {
        url: string;
        contentType?: string;
    } | null;
    /**
     * Returns the first detected `data` part of a chunk.
     * @returns The first `data` part detected in the chunk (if any).
     */
    data(): T | null;
    /**
     * Returns all tool request found in this chunk.
     * @returns Array of all tool request found in this chunk.
     */
    toolRequests(): ToolRequestPart[];
    /**
     * Attempts to extract the longest valid JSON substring from the accumulated chunks.
     * @returns The longest valid JSON substring found in the accumulated chunks.
     */
    output(): T | null;
    toJSON(): GenerateResponseChunkData;
}
declare function toGenerateRequest(options: GenerateOptions): Promise<GenerateRequest>;
interface GenerateOptions<O extends z.ZodTypeAny = z.ZodTypeAny, CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> {
    /** A model name (e.g. `vertexai/gemini-1.0-pro`) or reference. */
    model?: ModelArgument<CustomOptions>;
    /** The prompt for which to generate a response. Can be a string for a simple text prompt or one or more parts for multi-modal prompts. */
    prompt: string | Part | Part[];
    /** Retrieved documents to be used as context for this generation. */
    context?: DocumentData[];
    /** Conversation history for multi-turn prompting when supported by the underlying model. */
    history?: MessageData[];
    /** List of registered tool names or actions to treat as a tool for this generation if supported by the underlying model. */
    tools?: ToolArgument[];
    /** Number of candidate messages to generate. */
    candidates?: number;
    /** Configuration for the generation request. */
    config?: z.infer<CustomOptions>;
    /** Configuration for the desired output of the request. Defaults to the model's default output if unspecified. */
    output?: {
        format?: 'text' | 'json' | 'media';
        schema?: O;
        jsonSchema?: any;
    };
    /** When true, return tool calls for manual processing instead of automatically resolving them. */
    returnToolRequests?: boolean;
    /** When provided, models supporting streaming will call the provided callback with chunks as generation progresses. */
    streamingCallback?: StreamingCallback<GenerateResponseChunk>;
}
declare class NoValidCandidatesError extends GenkitError {
    detail: {
        response: GenerateResponse;
        [otherDetails: string]: any;
    };
    constructor({ message, response, detail, }: {
        message: string;
        response: GenerateResponse;
        detail?: Record<string, any>;
    });
}
/**
 * Generate calls a generative model based on the provided prompt and configuration. If
 * `history` is provided, the generation will include a conversation history in its
 * request. If `tools` are provided, the generate method will automatically resolve
 * tool calls returned from the model unless `returnToolRequests` is set to `true`.
 *
 * See `GenerateOptions` for detailed information about available options.
 *
 * @param options The options for this generation request.
 * @returns The generated response based on the provided parameters.
 */
declare function generate<O extends z.ZodTypeAny = z.ZodTypeAny, CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema>(options: GenerateOptions<O, CustomOptions> | PromiseLike<GenerateOptions<O, CustomOptions>>): Promise<GenerateResponse<z.infer<O>>>;
type GenerateStreamOptions<O extends z.ZodTypeAny = z.ZodTypeAny, CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema> = Omit<GenerateOptions<O, CustomOptions>, 'streamingCallback'>;
interface GenerateStreamResponse<O extends z.ZodTypeAny = z.ZodTypeAny> {
    stream: () => AsyncIterable<GenerateResponseChunkData>;
    response: () => Promise<GenerateResponse<O>>;
}
declare function generateStream<O extends z.ZodTypeAny = z.ZodTypeAny, CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema>(options: GenerateOptions<O, CustomOptions> | PromiseLike<GenerateOptions<O, CustomOptions>>): Promise<GenerateStreamResponse<O>>;

export { Candidate, type GenerateOptions, GenerateResponse, GenerateResponseChunk, type GenerateStreamOptions, type GenerateStreamResponse, Message, NoValidCandidatesError, generate, generateStream, toGenerateRequest };
