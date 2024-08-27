import { Mutex } from 'async-mutex';
import { TraceStore, TraceData, TraceQuery, TraceQueryResponse } from './types.mjs';
import 'zod';

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
 * Implementation of trace store that persists traces on local disk.
 */
declare class LocalFileTraceStore implements TraceStore {
    private readonly storeRoot;
    private mutexes;
    private filters;
    static defaultFilters: Record<string, string>;
    constructor(filters?: Record<string, string>);
    load(id: string): Promise<TraceData | undefined>;
    getMutex(id: string): Mutex;
    save(id: string, rawTrace: TraceData): Promise<void>;
    list(query?: TraceQuery): Promise<TraceQueryResponse>;
    private filter;
}

export { LocalFileTraceStore };
